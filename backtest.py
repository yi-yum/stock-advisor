import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


# ── 指標計算函式 ──────────────────────────────────────
def calculate_rsi(prices, period=14):
    """計算 RSI"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """計算 MACD"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def prepare_backtest_data(df: pd.DataFrame) -> pd.DataFrame:
    """準備回測數據，計算所有必需指標"""
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = calculate_rsi(df["Close"], period=14)
    df["MACD"], df["Signal"], df["Histogram"] = calculate_macd(df["Close"])

    # 計算金叉/死叉訊號
    df["MACD_Signal"] = 0
    df.loc[df["Histogram"] > 0, "MACD_Signal"] = 1  # 正值=金叉
    df.loc[df["Histogram"] < 0, "MACD_Signal"] = -1  # 負值=死叉

    # 計算均線交叉訊號
    df["MA_Cross"] = 0
    df.loc[df["MA20"] > df["MA50"], "MA_Cross"] = 1  # MA20在MA50上方
    df.loc[df["MA20"] < df["MA50"], "MA_Cross"] = -1  # MA20在MA50下方

    return df


# ── RSI 超賣策略 ──────────────────────────────────────
def backtest_rsi(df: pd.DataFrame, rsi_oversold: int = 30, rsi_overbought: int = 70, initial_capital: float = 100000.0) -> dict:
    """
    RSI 超賣策略回測
    - RSI < 超賣門檻 → 買入
    - RSI > 超買門檻 → 賣出
    """
    trades = []
    position = None  # {'buy_idx': int, 'buy_price': float, 'buy_date': datetime}
    equity_curve = [initial_capital]
    dates = [df.index[0]]

    for i in range(len(df)):
        row = df.iloc[i]
        close_price = float(row["Close"])
        current_date = row.name

        if pd.isna(row["RSI"]):
            continue

        rsi = float(row["RSI"])

        # 買入信號：RSI < 超賣門檻
        if position is None and rsi < rsi_oversold:
            position = {
                "buy_idx": i,
                "buy_price": close_price,
                "buy_date": current_date,
            }

        # 賣出信號：RSI > 超買門檻
        elif position is not None and rsi > rsi_overbought:
            sell_price = close_price
            sell_date = current_date
            return_pct = (sell_price - position["buy_price"]) / position["buy_price"] * 100

            trades.append({
                "buy_date": position["buy_date"],
                "buy_price": position["buy_price"],
                "sell_date": sell_date,
                "sell_price": sell_price,
                "return_pct": return_pct,
                "buy_days": (sell_date - position["buy_date"]).days,
            })

            # 更新資金曲線
            capital_after_trade = initial_capital * (1 + return_pct / 100)
            equity_curve.append(capital_after_trade)
            dates.append(sell_date)
            position = None

    # 計算績效指標
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t["return_pct"] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    if trades:
        avg_return = np.mean([t["return_pct"] for t in trades])
        total_return = (equity_curve[-1] - initial_capital) / initial_capital * 100
    else:
        avg_return = 0
        total_return = 0

    # 計算最大回撤
    max_drawdown = calculate_max_drawdown(equity_curve)

    # 計算 Sharpe Ratio
    sharpe = calculate_sharpe_ratio(equity_curve, dates)

    # 計算買入持有報酬
    buy_hold_return = (float(df.iloc[-1]["Close"]) - float(df.iloc[0]["Close"])) / float(df.iloc[0]["Close"]) * 100

    # 構建資金曲線 Series
    equity_series = pd.Series(equity_curve, index=dates if len(dates) == len(equity_curve) else pd.date_range(df.index[0], periods=len(equity_curve)))

    return {
        "trades": trades,
        "total_return": round(total_return, 2),
        "win_rate": round(win_rate, 2),
        "total_trades": total_trades,
        "avg_return": round(avg_return, 2),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe, 2),
        "equity_curve": equity_series,
        "buy_hold_return": round(buy_hold_return, 2),
        "strategy_name": "RSI 超賣策略",
    }


# ── MACD 金叉策略 ─────────────────────────────────────
def backtest_macd(df: pd.DataFrame, initial_capital: float = 100000.0) -> dict:
    """
    MACD 金叉策略回測
    - Histogram 由負轉正 → 買入
    - Histogram 由正轉負 → 賣出
    """
    trades = []
    position = None
    equity_curve = [initial_capital]
    dates = [df.index[0]]
    prev_signal = None

    for i in range(len(df)):
        row = df.iloc[i]
        close_price = float(row["Close"])
        current_date = row.name

        if pd.isna(row["Histogram"]):
            prev_signal = None
            continue

        histogram = float(row["Histogram"])

        # 確定當前信號
        current_signal = 1 if histogram > 0 else -1

        # 金叉訊號：由負轉正
        if position is None and prev_signal == -1 and current_signal == 1:
            position = {
                "buy_idx": i,
                "buy_price": close_price,
                "buy_date": current_date,
            }

        # 死叉訊號：由正轉負
        elif position is not None and prev_signal == 1 and current_signal == -1:
            sell_price = close_price
            sell_date = current_date
            return_pct = (sell_price - position["buy_price"]) / position["buy_price"] * 100

            trades.append({
                "buy_date": position["buy_date"],
                "buy_price": position["buy_price"],
                "sell_date": sell_date,
                "sell_price": sell_price,
                "return_pct": return_pct,
                "buy_days": (sell_date - position["buy_date"]).days,
            })

            capital_after_trade = initial_capital * (1 + return_pct / 100)
            equity_curve.append(capital_after_trade)
            dates.append(sell_date)
            position = None

        prev_signal = current_signal

    # 計算績效指標
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t["return_pct"] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    if trades:
        avg_return = np.mean([t["return_pct"] for t in trades])
        total_return = (equity_curve[-1] - initial_capital) / initial_capital * 100
    else:
        avg_return = 0
        total_return = 0

    max_drawdown = calculate_max_drawdown(equity_curve)
    sharpe = calculate_sharpe_ratio(equity_curve, dates)
    buy_hold_return = (float(df.iloc[-1]["Close"]) - float(df.iloc[0]["Close"])) / float(df.iloc[0]["Close"]) * 100

    equity_series = pd.Series(equity_curve, index=dates if len(dates) == len(equity_curve) else pd.date_range(df.index[0], periods=len(equity_curve)))

    return {
        "trades": trades,
        "total_return": round(total_return, 2),
        "win_rate": round(win_rate, 2),
        "total_trades": total_trades,
        "avg_return": round(avg_return, 2),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe, 2),
        "equity_curve": equity_series,
        "buy_hold_return": round(buy_hold_return, 2),
        "strategy_name": "MACD 金叉策略",
    }


# ── 均線交叉策略 ──────────────────────────────────────
def backtest_ma_cross(df: pd.DataFrame, initial_capital: float = 100000.0) -> dict:
    """
    均線交叉策略回測
    - MA20 向上穿越 MA50 → 買入
    - MA20 向下穿越 MA50 → 賣出
    """
    trades = []
    position = None
    equity_curve = [initial_capital]
    dates = [df.index[0]]
    prev_signal = None

    for i in range(len(df)):
        row = df.iloc[i]
        close_price = float(row["Close"])
        current_date = row.name

        if pd.isna(row["MA20"]) or pd.isna(row["MA50"]):
            prev_signal = None
            continue

        ma20 = float(row["MA20"])
        ma50 = float(row["MA50"])

        # 確定當前信號
        current_signal = 1 if ma20 > ma50 else -1

        # 金叉訊號：MA20 向上穿越 MA50
        if position is None and prev_signal == -1 and current_signal == 1:
            position = {
                "buy_idx": i,
                "buy_price": close_price,
                "buy_date": current_date,
            }

        # 死叉訊號：MA20 向下穿越 MA50
        elif position is not None and prev_signal == 1 and current_signal == -1:
            sell_price = close_price
            sell_date = current_date
            return_pct = (sell_price - position["buy_price"]) / position["buy_price"] * 100

            trades.append({
                "buy_date": position["buy_date"],
                "buy_price": position["buy_price"],
                "sell_date": sell_date,
                "sell_price": sell_price,
                "return_pct": return_pct,
                "buy_days": (sell_date - position["buy_date"]).days,
            })

            capital_after_trade = initial_capital * (1 + return_pct / 100)
            equity_curve.append(capital_after_trade)
            dates.append(sell_date)
            position = None

        prev_signal = current_signal

    # 計算績效指標
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t["return_pct"] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    if trades:
        avg_return = np.mean([t["return_pct"] for t in trades])
        total_return = (equity_curve[-1] - initial_capital) / initial_capital * 100
    else:
        avg_return = 0
        total_return = 0

    max_drawdown = calculate_max_drawdown(equity_curve)
    sharpe = calculate_sharpe_ratio(equity_curve, dates)
    buy_hold_return = (float(df.iloc[-1]["Close"]) - float(df.iloc[0]["Close"])) / float(df.iloc[0]["Close"]) * 100

    equity_series = pd.Series(equity_curve, index=dates if len(dates) == len(equity_curve) else pd.date_range(df.index[0], periods=len(equity_curve)))

    return {
        "trades": trades,
        "total_return": round(total_return, 2),
        "win_rate": round(win_rate, 2),
        "total_trades": total_trades,
        "avg_return": round(avg_return, 2),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe, 2),
        "equity_curve": equity_series,
        "buy_hold_return": round(buy_hold_return, 2),
        "strategy_name": "均線交叉策略",
    }


# ── 輔助函式 ──────────────────────────────────────────
def calculate_max_drawdown(equity_curve: list) -> float:
    """計算最大回撤"""
    if len(equity_curve) < 2:
        return 0.0

    equity_array = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max * 100
    return float(np.min(drawdown))


def calculate_sharpe_ratio(equity_curve: list, dates: list, risk_free_rate: float = 0.03) -> float:
    """計算年化 Sharpe Ratio"""
    if len(equity_curve) < 2:
        return 0.0

    # 計算日報酬率
    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])

    # 年化報酬率
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    days = (dates[-1] - dates[0]).days if len(dates) > 1 else 1
    annualized_return = (1 + total_return) ** (365 / max(days, 1)) - 1

    # 日標準差 -> 年化標準差
    daily_std = np.std(returns) if len(returns) > 0 else 0
    annualized_std = daily_std * np.sqrt(252)

    # Sharpe Ratio
    if annualized_std == 0:
        return 0.0

    sharpe = (annualized_return - risk_free_rate) / annualized_std
    return float(sharpe)


def backtest_structural_pullback(df: pd.DataFrame, initial_capital: float = 100000.0) -> dict:
    """
    結構性回調策略回測（與每日掃描一致的最新版本）

    進場條件（三層過濾）：
      Layer1 — 趨勢：EMA20>EMA50 且 ADX>20
      Layer2 — 回調：距EMA20 -8%~+2%，RSI 40-62，縮量（VolumeRatio<1）
      Layer3 — 轉強：MACD柱體開始回升 OR 收紅K，至少滿足一項

    出場條件（先到先出）：
      A. 收盤跌破止損（max(EMA50*0.98, Low20*0.99, entry-2*ATR)）
      B. EMA20 < EMA50（多頭趨勢破壞）
    """
    # ── 計算所有指標 ──────────────────────────────────────
    df = df.copy()
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]
    open_  = df["Open"] if "Open" in df.columns else close

    # EMA
    df["EMA20"]  = close.ewm(span=20,  adjust=False).mean()
    df["EMA50"]  = close.ewm(span=50,  adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()

    # RSI(14)
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0.0)).ewm(span=14, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # MACD 柱體
    macd  = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    sig   = macd.ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = macd - sig

    # ATR(14)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=14, adjust=False).mean()

    # ADX
    up   = high.diff()
    down = -low.diff()
    dm_p = up.where((up > down) & (up > 0), 0.0)
    dm_m = down.where((down > up) & (down > 0), 0.0)
    atr_s = tr.ewm(span=14, adjust=False).mean()
    di_p  = 100 * dm_p.ewm(span=14, adjust=False).mean() / (atr_s + 1e-10)
    di_m  = 100 * dm_m.ewm(span=14, adjust=False).mean() / (atr_s + 1e-10)
    denom = (di_p + di_m).replace(0, float("nan"))
    df["ADX"] = (100 * (di_p - di_m).abs() / denom).ewm(span=14, adjust=False).mean()

    # Volume ratio
    vol_ma20 = volume.rolling(20).mean()
    df["VolRatio"] = volume / (vol_ma20 + 1e-10)

    # 20日高低（避免 look-ahead）
    df["Low20"]  = close.rolling(20).min().shift(1)

    df.dropna(inplace=True)

    # ── 回測主迴圈 ──────────────────────────────────────
    trades       = []
    position     = None
    equity       = initial_capital
    equity_curve = [equity]
    dates        = [df.index[0]]

    for i in range(1, len(df)):
        row      = df.iloc[i]
        prev_row = df.iloc[i - 1]
        c        = float(row["Close"])
        o        = float(open_.iloc[i]) if "Open" in df.columns else c
        ema20    = float(row["EMA20"])
        ema50    = float(row["EMA50"])
        rsi      = float(row["RSI"])
        adx      = float(row["ADX"])
        vol_r    = float(row["VolRatio"])
        atr      = float(row["ATR"])
        hist     = float(row["MACD_hist"])
        prev_hist = float(prev_row["MACD_hist"])
        low20    = float(row["Low20"]) if not pd.isna(row["Low20"]) else c * 0.95

        if position is not None:
            # ── 出場條件 ──────────────────────────────
            sl = position["sl"]
            if c < sl:
                # A. 跌破止損
                _record_trade(trades, position, c, row.name, "止損出場")
                equity = equity * (1 + (c - position["entry"]) / position["entry"])
                equity_curve.append(equity)
                dates.append(row.name)
                position = None
            elif ema20 < ema50:
                # B. 趨勢破壞
                _record_trade(trades, position, c, row.name, "趨勢破壞")
                equity = equity * (1 + (c - position["entry"]) / position["entry"])
                equity_curve.append(equity)
                dates.append(row.name)
                position = None

        if position is None:
            # ── 進場條件 ──────────────────────────────
            # Layer1: 趨勢
            trend_ok = (ema20 > ema50) and (adx > 20)
            if not trend_ok:
                continue

            # Layer2: 回調品質
            dist = (c - ema20) / ema20 * 100
            pullback_ok = (-8.0 <= dist <= 2.0) and (40 <= rsi <= 62) and (vol_r < 1.0)
            if not pullback_ok:
                continue

            # Layer3: 轉強訊號（至少一項）
            macd_rising = hist > prev_hist
            is_green    = c > o
            turn_ok = macd_rising or is_green
            if not turn_ok:
                continue

            # 計算止損
            sl_raw = max(ema50 * 0.98, low20 * 0.99, c - 2.0 * atr)
            risk   = c - sl_raw
            if risk <= 0:
                continue

            position = {
                "entry":      c,
                "entry_date": row.name,
                "sl":         sl_raw,
                "tp":         c + 2.5 * risk,
            }

    # 若回測結束時仍持倉，按最後收盤平倉
    if position is not None:
        last = df.iloc[-1]
        _record_trade(trades, position, float(last["Close"]), last.name, "回測結束平倉")
        equity = equity * (1 + (float(last["Close"]) - position["entry"]) / position["entry"])
        equity_curve.append(equity)
        dates.append(last.name)

    # ── 績效計算 ──────────────────────────────────────
    total_trades   = len(trades)
    wins           = [t for t in trades if t["return_pct"] > 0]
    win_rate       = len(wins) / total_trades if total_trades else 0
    avg_return     = np.mean([t["return_pct"] for t in trades]) if trades else 0
    total_return   = (equity - initial_capital) / initial_capital * 100
    max_dd         = calculate_max_drawdown(equity_curve)
    equity_series  = pd.Series(equity_curve, index=dates if len(dates) == len(equity_curve) else pd.date_range(df.index[0], periods=len(equity_curve)))
    sharpe         = calculate_sharpe_ratio(equity_series, dates)
    buy_hold       = (float(df["Close"].iloc[-1]) - float(df["Close"].iloc[0])) / float(df["Close"].iloc[0]) * 100

    return {
        "trades":          trades,
        "total_return":    round(total_return, 2),
        "win_rate":        round(win_rate, 2),
        "total_trades":    total_trades,
        "avg_return":      round(avg_return, 2),
        "max_drawdown":    round(max_dd, 2),
        "sharpe_ratio":    round(sharpe, 2),
        "equity_curve":    equity_series,
        "buy_hold_return": round(buy_hold, 2),
        "strategy_name":   "結構性回調策略",
    }


def _record_trade(trades: list, position: dict, exit_price: float, exit_date, reason: str):
    ret = (exit_price - position["entry"]) / position["entry"] * 100
    trades.append({
        "buy_date":   position["entry_date"],
        "buy_price":  position["entry"],
        "sell_date":  exit_date,
        "sell_price": exit_price,
        "return_pct": round(ret, 2),
        "buy_days":   (exit_date - position["entry_date"]).days if hasattr(exit_date, '__sub__') else 0,
        "exit_reason": reason,
    })


def run_backtest(
    df: pd.DataFrame,
    strategy: str,
    rsi_oversold: int = 30,
    rsi_overbought: int = 70,
    initial_capital: float = 100000.0,
) -> dict:
    """
    通用回測函式

    Args:
        df: 已含指標的日線 DataFrame
        strategy: "rsi" | "macd" | "ma_cross" | "structural_pullback"
        rsi_oversold: RSI 超賣門檻（預設30）
        rsi_overbought: RSI 超買門檻（預設70）
        initial_capital: 初始資本（預設100000）

    Returns:
        回測結果字典
    """
    if strategy == "structural_pullback":
        return backtest_structural_pullback(df, initial_capital)

    df = prepare_backtest_data(df)

    if strategy == "rsi":
        return backtest_rsi(df, rsi_oversold, rsi_overbought, initial_capital)
    elif strategy == "macd":
        return backtest_macd(df, initial_capital)
    elif strategy == "ma_cross":
        return backtest_ma_cross(df, initial_capital)
    else:
        raise ValueError(f"未知策略: {strategy}")


# ── Streamlit UI ──────────────────────────────────────
def render_backtest_tab():
    """Streamlit 回測頁面"""
    st.markdown("## 🎯 簡易回測模組")

    with st.expander("📋 使用說明", expanded=False):
        st.write("""
        **支援策略：**
        1. **RSI 超賣策略** - RSI 低於超賣門檻時買入，高於超買門檻時賣出
        2. **MACD 金叉策略** - MACD Histogram 由負轉正買入，由正轉負賣出
        3. **均線交叉策略** - MA20 向上穿越 MA50 買入，向下穿越時賣出

        **重要提示：**
        - 此回測不代表未來表現，僅供策略驗證參考
        - 假設每次交易投入全部資金（全倉）
        - 不考慮手續費
        """)

    col1, col2 = st.columns(2)

    with col1:
        symbol = st.text_input("📈 股票代號", value="AAPL", placeholder="如：AAPL、2330.TW、BTC-USD")

    with col2:
        strategy = st.selectbox(
            "🎯 選擇策略",
            options=["rsi", "macd", "ma_cross"],
            format_func=lambda x: {
                "rsi": "RSI 超賣策略",
                "macd": "MACD 金叉策略",
                "ma_cross": "均線交叉策略",
            }.get(x, x),
        )

    col3, col4 = st.columns(2)

    with col3:
        period = st.selectbox(
            "📅 回測期間",
            options=["1y", "2y", "3y"],
            format_func=lambda x: {
                "1y": "1年",
                "2y": "2年",
                "3y": "3年",
            }.get(x, x),
        )

    with col4:
        initial_capital = st.number_input(
            "💰 初始資本",
            value=100000.0,
            min_value=10000.0,
            step=10000.0,
            format="%.0f",
        )

    # 策略參數
    if strategy == "rsi":
        col5, col6 = st.columns(2)
        with col5:
            rsi_oversold = st.slider("RSI 超賣門檻", 10, 30, 30)
        with col6:
            rsi_overbought = st.slider("RSI 超買門檻", 70, 90, 70)
    else:
        rsi_oversold = 30
        rsi_overbought = 70

    if st.button("▶️ 開始回測", use_container_width=True):
        try:
            with st.spinner(f"正在取得 {symbol} 的數據並執行回測..."):
                # 取得數據
                ticker = yf.Ticker(symbol.upper())
                df = ticker.history(period=period)

                if df.empty:
                    st.error(f"❌ 無法取得 {symbol} 的數據，請確認代號是否正確")
                    return

                # 執行回測
                result = run_backtest(
                    df,
                    strategy,
                    rsi_oversold=rsi_oversold,
                    rsi_overbought=rsi_overbought,
                    initial_capital=initial_capital,
                )

                # ── 顯示核心指標卡 ────────────────────────
                st.markdown(f"### 📊 {result['strategy_name']} - 回測結果")

                metric_cols = st.columns(5)

                with metric_cols[0]:
                    st.metric(
                        "總報酬率",
                        f"{result['total_return']:.2f}%",
                        delta=f"vs 買入持有: {result['buy_hold_return']:.2f}%",
                        delta_color="off",
                    )

                with metric_cols[1]:
                    st.metric(
                        "勝率",
                        f"{result['win_rate']*100:.1f}%",
                        delta=f"{result['total_trades']} 筆交易",
                        delta_color="off",
                    )

                with metric_cols[2]:
                    st.metric(
                        "平均報酬",
                        f"{result['avg_return']:.2f}%",
                        delta_color="off",
                    )

                with metric_cols[3]:
                    color = "normal" if result['max_drawdown'] > -10 else "inverse"
                    st.metric(
                        "最大回撤",
                        f"{result['max_drawdown']:.2f}%",
                        delta_color=color,
                    )

                with metric_cols[4]:
                    st.metric(
                        "Sharpe Ratio",
                        f"{result['sharpe_ratio']:.2f}",
                        delta_color="off",
                    )

                # ── 資金曲線圖 ────────────────────────────
                st.markdown("### 💹 資金曲線對比")

                fig = go.Figure()

                # 策略資金曲線
                fig.add_trace(go.Scatter(
                    x=result['equity_curve'].index,
                    y=result['equity_curve'].values,
                    name=result['strategy_name'],
                    line=dict(color='#00D084', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0, 208, 132, 0.2)',
                ))

                # 買入持有對比線
                buy_hold_curve = initial_capital * (1 + (df['Close'] / df['Close'].iloc[0] - 1))
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=buy_hold_curve.values,
                    name="買入持有",
                    line=dict(color='#0066CC', width=2, dash='dash'),
                ))

                fig.update_layout(
                    title="策略資金曲線 vs 買入持有",
                    xaxis_title="日期",
                    yaxis_title="資金（¥）",
                    template="plotly_white",
                    height=450,
                    hovermode='x unified',
                    legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)'),
                )

                st.plotly_chart(fig, use_container_width=True)

                # ── 交易明細表格 ────────────────────────
                st.markdown("### 📋 交易明細")

                if result['trades']:
                    trades_df = pd.DataFrame(result['trades'])
                    trades_df['buy_date'] = trades_df['buy_date'].dt.strftime('%Y-%m-%d')
                    trades_df['sell_date'] = trades_df['sell_date'].dt.strftime('%Y-%m-%d')
                    trades_df['buy_price'] = trades_df['buy_price'].apply(lambda x: f"${x:.2f}")
                    trades_df['sell_price'] = trades_df['sell_price'].apply(lambda x: f"${x:.2f}")
                    trades_df['return_pct'] = trades_df['return_pct'].apply(lambda x: f"{x:.2f}%")

                    trades_df = trades_df.rename(columns={
                        'buy_date': '買入日期',
                        'buy_price': '買入價',
                        'sell_date': '賣出日期',
                        'sell_price': '賣出價',
                        'return_pct': '報酬%',
                        'buy_days': '持倉天數',
                    })

                    st.dataframe(
                        trades_df[['買入日期', '買入價', '賣出日期', '賣出價', '報酬%', '持倉天數']],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.warning("⚠️ 回測期間內無交易訊號")

                # ── 警示 ──────────────────────────────────
                st.warning(
                    "⚠️ **重要提示**\n"
                    "此回測不代表未來表現，僅供策略驗證參考。過往績效不代表未來結果。"
                )

        except Exception as e:
            st.error(f"❌ 回測時發生錯誤：{str(e)}")
            st.write("可能原因：")
            st.write("- 股票代號不正確")
            st.write("- 該股票數據不足")
            st.write("- 網路連接問題")


if __name__ == "__main__":
    render_backtest_tab()
