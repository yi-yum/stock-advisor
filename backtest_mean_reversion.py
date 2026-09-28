"""
backtest_mean_reversion.py — 淡季均值回歸策略回測
=================================================
只在 5–9 月操作
進場：RSI(14) < RSI_ENTRY（超賣）→ 次日開盤買
出場（擇一）：
  1. RSI(14) > RSI_EXIT（回歸超買）
  2. 跌破 STOP_LOSS
  3. 持倉超過 MAX_DAYS 天
"""

import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_portfolio import load_ohlcv, get_tw_symbols, get_sp500_symbols, print_summary

# ── 參數 ──────────────────────────────────────────────────────────────────────
RSI_ENTRY  = 35      # 進場：RSI 低於此值
RSI_EXIT   = 55      # 出場：RSI 回升超過此值
STOP_LOSS  = -0.08   # 止損：-8%
MAX_DAYS   = 20      # 最長持倉天數
WEAK_MONTHS = {5, 6, 7, 8, 9}

TW_FEE_BUY  = 0.001425
TW_FEE_SELL = 0.001425 + 0.003


def calc_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    rs  = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def simulate_mr(symbol: str, df: pd.DataFrame,
                fee_buy: float, fee_sell: float,
                end_date: str) -> list[dict]:
    if len(df) < 30:
        return []

    closes = df["Close"].values.astype(float)
    opens  = df["Open"].values.astype(float)
    dates  = df.index
    rsi    = calc_rsi(closes)
    n      = len(df)
    end_ts = pd.Timestamp(end_date)

    trades    = []
    in_trade  = False
    entry_date = entry_price = entry_idx = None

    for i in range(14, n - 1):
        if dates[i] > end_ts:
            break

        month = dates[i].month

        if not in_trade:
            # 只在淡季找進場
            if month not in WEAK_MONTHS:
                continue
            if rsi[i] < RSI_ENTRY:
                # 次日開盤進場
                entry_idx   = i + 1
                if entry_idx >= n:
                    break
                entry_price = opens[entry_idx] * (1 + fee_buy)
                entry_date  = dates[entry_idx]
                in_trade    = True
        else:
            days_held = i - entry_idx
            cur_price = closes[i]
            cur_rsi   = rsi[i]
            ret_raw   = (cur_price - opens[entry_idx]) / opens[entry_idx]

            exit_reason = None
            if cur_rsi > RSI_EXIT:
                exit_reason = "RSI回歸"
            elif ret_raw <= STOP_LOSS:
                exit_reason = "止損"
            elif days_held >= MAX_DAYS:
                exit_reason = "到期"

            if exit_reason:
                exit_price = opens[i + 1] * (1 - fee_sell) if i + 1 < n else closes[i] * (1 - fee_sell)
                ret_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    "symbol":       symbol,
                    "entry_date":   entry_date.strftime("%Y-%m-%d"),
                    "entry_price":  round(entry_price, 4),
                    "exit_date":    dates[i].strftime("%Y-%m-%d"),
                    "exit_price":   round(exit_price, 4),
                    "return_pct":   round(ret_pct, 4),
                    "days_held":    days_held,
                    "exit_reason":  exit_reason,
                    "entry_rsi":    round(rsi[entry_idx], 1),
                    "entry_month":  entry_date.month,
                    "year":         entry_date.year,
                })
                in_trade  = False
                entry_date = entry_price = entry_idx = None

    return trades


def main():
    import argparse
    parser = argparse.ArgumentParser(description="淡季均值回歸回測")
    parser.add_argument("--market",  default="tw",      choices=["tw", "us"])
    parser.add_argument("--start",   default="2021-01-01")
    parser.add_argument("--end",     default="2026-09-22")
    parser.add_argument("--rsi-entry", type=float, default=RSI_ENTRY)
    parser.add_argument("--rsi-exit",  type=float, default=RSI_EXIT)
    parser.add_argument("--stop",      type=float, default=abs(STOP_LOSS))
    parser.add_argument("--max-days",  type=int,   default=MAX_DAYS)
    args = parser.parse_args()

    rsi_entry = args.rsi_entry
    rsi_exit  = args.rsi_exit
    stop_loss = -abs(args.stop)
    max_days  = args.max_days

    print(f"均值回歸策略（淡季 5-9 月）")
    print(f"進場 RSI < {rsi_entry}，出場 RSI > {rsi_exit}，止損 {stop_loss*100:.0f}%，最長 {max_days} 天")
    print(f"市場：{args.market.upper()}，期間：{args.start} ~ {args.end}")

    if args.market == "tw":
        symbols  = get_tw_symbols()
        fee_buy  = TW_FEE_BUY
        fee_sell = TW_FEE_SELL
    else:
        symbols  = get_sp500_symbols()
        fee_buy  = fee_sell = 0.0

    print(f"股票清單：{len(symbols)} 支（使用現有快取）\n")

    all_trades  = []
    valid_count = 0
    t0 = time.time()

    for i, sym in enumerate(symbols):
        if (i + 1) % 200 == 0:
            print(f"  進度 {i+1}/{len(symbols)}，訊號累計 {len(all_trades)} 筆")
        df = load_ohlcv(sym, args.start, args.end)
        if df is None or len(df) < 30:
            continue
        valid_count += 1
        trades = simulate_mr(sym, df, fee_buy, fee_sell, args.end)
        all_trades.extend(trades)

    print_summary(all_trades, args.start, args.end, len(symbols), valid_count)

    # 出場原因分佈
    if all_trades:
        df_t = pd.DataFrame(all_trades)
        print("\n出場原因：")
        for reason, g in df_t.groupby("exit_reason"):
            wr = (g["return_pct"] > 0).sum() / len(g) * 100
            ev = g["return_pct"].mean()
            print(f"  {reason}: n={len(g)}, 勝率={wr:.1f}%, EV={ev:+.2f}%")

        print("\n進場 RSI 分層：")
        for lo, hi in [(0,25),(25,30),(30,35)]:
            d = df_t[(df_t["entry_rsi"]>=lo) & (df_t["entry_rsi"]<hi)]
            if len(d) > 0:
                wr = (d["return_pct"]>0).sum()/len(d)*100
                ev = d["return_pct"].mean()
                print(f"  RSI {lo}-{hi}: n={len(d)}, 勝率={wr:.1f}%, EV={ev:+.2f}%")

        today_str = date.today().strftime("%Y%m%d")
        csv_path = Path("backtest_results") / f"signals_mr_{args.market}_{today_str}.csv"
        df_t.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\nCSV 已儲存：{csv_path}")


if __name__ == "__main__":
    main()
