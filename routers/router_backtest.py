import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from backtest import run_backtest

router = APIRouter(prefix="/api", tags=["backtest"])


class BacktestRequest(BaseModel):
    symbol: str
    strategy: str  # "rsi" | "macd" | "ma_cross"
    period_years: int = 2
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    initial_capital: float = 100000.0


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


def serialize_equity_curve(equity_series):
    """將 pd.Series 轉換為 JSON-safe 格式"""
    if isinstance(equity_series, pd.Series):
        df = equity_series.reset_index()
        df.columns = ['date', 'value']
        df['date'] = df['date'].astype(str)
        # 將 NaN 轉為 None
        df = df.where(pd.notna(df), None)
        return df.to_dict(orient='records')
    return []


def serialize_trades(trades):
    """將交易列表中的日期轉換為字串"""
    serialized = []
    for trade in trades:
        trade_copy = trade.copy()
        # 轉換日期欄位
        if 'buy_date' in trade_copy and hasattr(trade_copy['buy_date'], 'strftime'):
            trade_copy['buy_date'] = trade_copy['buy_date'].strftime('%Y-%m-%d')
        if 'sell_date' in trade_copy and hasattr(trade_copy['sell_date'], 'strftime'):
            trade_copy['sell_date'] = trade_copy['sell_date'].strftime('%Y-%m-%d')
        serialized.append(trade_copy)
    return serialized


@router.post("/backtest")
async def backtest(request: BacktestRequest):
    """
    執行回測

    Args:
        symbol: 股票代號
        strategy: 回測策略 ("rsi", "macd", "ma_cross")
        period_years: 回測年數（預設2年）
        rsi_oversold: RSI 超賣門檻（預設30）
        rsi_overbought: RSI 超買門檻（預設70）
        initial_capital: 初始資本（預設100000）

    Returns:
        JSON: 回測結果（已序列化）
    """
    try:
        # 驗證策略參數
        SUPPORTED = ["rsi", "macd", "ma_cross", "structural_pullback"]
        if request.strategy not in SUPPORTED:
            raise HTTPException(
                status_code=400,
                detail=f"不支援的策略: {request.strategy}。支援: {', '.join(SUPPORTED)}"
            )

        # 下載數據
        try:
            ticker = yf.Ticker(request.symbol)
            df = ticker.history(period=f"{request.period_years}y", interval="1d")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"無法取得 {request.symbol} 的數據: {str(e)}"
            )

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail=f"無法取得 {request.symbol} 的有效數據"
            )

        # 準備數據，計算指標
        df = df.copy()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        df["RSI"] = calculate_rsi(df["Close"], period=14)
        df["MACD"], df["Signal"], df["Histogram"] = calculate_macd(df["Close"])

        # 計算金叉/死叉訊號
        df["MACD_Signal"] = 0
        df.loc[df["Histogram"] > 0, "MACD_Signal"] = 1
        df.loc[df["Histogram"] < 0, "MACD_Signal"] = -1

        # 計算均線交叉訊號
        df["MA_Cross"] = 0
        df.loc[df["MA20"] > df["MA50"], "MA_Cross"] = 1
        df.loc[df["MA20"] < df["MA50"], "MA_Cross"] = -1

        # 執行回測
        result = run_backtest(
            df,
            request.strategy,
            rsi_oversold=request.rsi_oversold,
            rsi_overbought=request.rsi_overbought,
            initial_capital=request.initial_capital
        )

        # 序列化結果
        serialized_result = {
            "symbol": request.symbol,
            "strategy": request.strategy,
            "period_years": request.period_years,
            "initial_capital": request.initial_capital,
            "total_return": result["total_return"],
            "win_rate": result["win_rate"],
            "total_trades": result["total_trades"],
            "avg_return": result["avg_return"],
            "max_drawdown": result["max_drawdown"],
            "sharpe_ratio": result["sharpe_ratio"],
            "buy_hold_return": result["buy_hold_return"],
            "strategy_name": result["strategy_name"],
            "equity_curve": serialize_equity_curve(result["equity_curve"]),
            "trades": serialize_trades(result["trades"])
        }

        return serialized_result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"回測發生錯誤: {str(e)}")
