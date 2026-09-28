"""
backtest_pullback_portfolio.py — 結構性回調（Structural Pullback）組合回測
==========================================================================
目的：與 backtest_portfolio.py（三重 SuperTrend）用同一套方法論（次日開盤成交、
      同樣手續費、同樣大盤/季節過濾）回測結構性回調策略，才能公平比較兩者績效，
      並找出結構性回調「在什麼條件下」可能贏過三重 SuperTrend（如果有的話）。

進場：signal_engine.get_signal() 的三層邏輯（趨勢≥4 + 回調有效 + entry_score≥7 + RR≥1.5）
     → 訊號日收盤確認 → 次日開盤買入
出場：EMA20 跌破 EMA50（趨勢轉弱）或收盤跌破 SL → 次日開盤賣出
     回測結束仍持倉 → 最後一日收盤結算

複用 backtest_portfolio.py 的股票清單、快取下載、大盤狀態、手續費設定，
避免重新造輪子（也避免兩份回測的手續費/清單邏輯各自漂移）。

執行：python backtest_pullback_portfolio.py
     python backtest_pullback_portfolio.py --taiex-min 3 --season
     python backtest_pullback_portfolio.py --no-download   （用現有快取，不重抓）
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from signal_engine import calc_indicators
from backtest_portfolio import (
    get_tw_symbols, get_us_symbols, load_ohlcv, batch_download,
    load_taiex_state, TW_FEE_BUY, TW_FEE_SELL, AVOID_MONTHS,
    CACHE_DIR, RESULT_DIR,
)

REQUIRED_COLS = ["MA20", "MA50", "EMA20", "EMA50", "RSI", "ATR", "ADX"]


def simulate_pullback_trades(symbol: str, df: pd.DataFrame,
                             fee_buy: float, fee_sell: float,
                             taiex_lookup: dict | None = None,
                             taiex_min: int = 0,
                             season_filter: bool = False) -> list[dict]:
    """對單支股票跑結構性回調回測（次日開盤成交，與三重ST回測方法論一致）。"""
    if len(df) < 260:   # EMA200 需要暖機，資料太短直接跳過
        return []

    calc = calc_indicators(df.copy())
    calc.dropna(subset=REQUIRED_COLS, inplace=True)
    if len(calc) < 60:
        return []

    dates  = calc.index
    opens  = calc["Open"].values.astype(float)
    highs  = calc["High"].values.astype(float)
    lows   = calc["Low"].values.astype(float)
    closes = calc["Close"].values.astype(float)
    e20    = calc["EMA20"].values.astype(float)
    e50    = calc["EMA50"].values.astype(float)
    e200   = calc["EMA200"].values.astype(float)
    macd   = calc["MACD_hist"].values.astype(float)
    rsi    = calc["RSI"].values.astype(float)
    vr     = calc["VolumeRatio"].values.astype(float)
    atr    = calc["ATR"].values.astype(float)
    di_p   = calc["DI_P"].values.astype(float)
    di_m   = calc["DI_M"].values.astype(float)
    low20  = calc["Low20"].values.astype(float)
    n      = len(calc)

    trades = []
    position = None

    for i in range(2, n):
        date_str = dates[i].strftime("%Y-%m-%d")

        # ── 出場（次日開盤）─────────────────────────────────────────
        if position is not None:
            hit_sl      = closes[i] <= position["sl"]
            trend_break = e20[i] < e50[i]
            if hit_sl or trend_break:
                exec_idx = i + 1
                if exec_idx >= n:
                    exec_idx = i
                    xp = closes[exec_idx]
                else:
                    xp = opens[exec_idx]
                if not (np.isnan(xp) or xp <= 0):
                    exit_price = xp * (1 - fee_sell)
                    ret = (exit_price - position["entry_price"]) / position["entry_price"]
                    trades.append({
                        "symbol":      symbol,
                        "signal_date": dates[i].date(),
                        "entry_date":  position["entry_date"],
                        "entry_price": round(position["entry_price"], 4),
                        "exit_date":   dates[exec_idx].date(),
                        "exit_price":  round(exit_price, 4),
                        "return_pct":  round(ret * 100, 4),
                        "days_held":   (dates[exec_idx] - position["entry_date"]).days,
                        "exit_reason": "止損出場" if hit_sl else "趨勢出場",
                        "year":        position["entry_date"].year,
                        "entry_score": position["entry_score"],
                        "trend_score": position["trend_score"],
                        "taiex_green": position["taiex_green"],
                    })
                    position = None
                    continue

        # ── 進場（次日開盤）─────────────────────────────────────────
        if position is None:
            ts = 0
            if e20[i] > e50[i]:                              ts += 2
            if not np.isnan(e200[i]) and e50[i] > e200[i]:   ts += 2
            if ts < 4:
                continue

            c = closes[i]
            dist = (c - e20[i]) / e20[i] * 100
            if not (-8.0 <= dist <= 2.0):
                continue
            if not (40 <= rsi[i] <= 62):
                continue
            if not (vr[i] < 1.0):
                continue

            es = 0
            if macd[i] > macd[i-1] > macd[i-2]: es += 2
            elif macd[i] > macd[i-1]:            es += 1
            is_green = c > opens[i]
            if is_green:
                es += 2
                if vr[i] > 1.0 and vr[i-1] < 0.9: es += 1
            if rsi[i] > 50 and rsi[i-1] <= 50:  es += 2
            elif rsi[i] > 50:                    es += 1
            dr = highs[i] - lows[i]
            if dr > 1e-6 and (c - lows[i]) / dr >= 0.8: es += 1
            if di_p[i] > di_m[i]:                es += 1

            if es < 7:
                continue

            l20v   = low20[i] if not np.isnan(low20[i]) else c * 0.92
            sl_raw = max(e50[i] * 0.98, l20v * 0.99, c - 2.0 * atr[i])
            risk   = c - sl_raw
            if risk <= 0:
                continue
            actual_rr = (c + 2.5 * risk - c) / risk
            if actual_rr < 1.5:
                continue

            # 季節過濾
            if season_filter and dates[i].month in AVOID_MONTHS:
                continue

            # 大盤過濾
            if taiex_min > 0 and taiex_lookup:
                tg = taiex_lookup.get(date_str)
                if tg is None:
                    tg = next(
                        (taiex_lookup[k] for k in sorted(taiex_lookup.keys(), reverse=True) if k <= date_str),
                        0,
                    )
                if tg < taiex_min:
                    continue

            exec_idx = i + 1
            if exec_idx >= n:
                break
            ep = opens[exec_idx]
            if np.isnan(ep) or ep <= 0:
                continue

            position = {
                "entry_price": ep * (1 + fee_buy),
                "entry_date":  dates[exec_idx],
                "sl":          sl_raw,
                "entry_score": es,
                "trend_score": ts,
                "taiex_green": taiex_lookup.get(date_str) if taiex_lookup else None,
            }

    # 回測結束仍持倉 → 最後收盤結算
    if position is not None:
        c = closes[-1]
        d = dates[-1]
        ret = (c - position["entry_price"]) / position["entry_price"]
        trades.append({
            "symbol":      symbol,
            "signal_date": d.date(),
            "entry_date":  position["entry_date"],
            "entry_price": round(position["entry_price"], 4),
            "exit_date":   d.date(),
            "exit_price":  round(c, 4),
            "return_pct":  round(ret * 100, 4),
            "days_held":   (d - position["entry_date"]).days,
            "exit_reason": "回測結束",
            "year":        position["entry_date"].year,
            "entry_score": position["entry_score"],
            "trend_score": position["trend_score"],
            "taiex_green": position["taiex_green"],
        })

    return trades


def print_summary(trades: list[dict], start: str, end: str, total_symbols: int, valid_symbols: int):
    df = pd.DataFrame(trades)
    print("\n" + "=" * 50)
    print("  結構性回調（Structural Pullback）組合回測結果")
    print("=" * 50)
    print(f"期間：{start} ~ {end}")
    print(f"股票池：{total_symbols} 支（有效 {valid_symbols} 支）")
    if df.empty:
        print("\n無任何交易紀錄（回測期間內完全沒有符合條件的訊號）。")
        print("=" * 50)
        return

    wins   = df[df["return_pct"] > 0]
    total  = len(df)
    wr     = len(wins) / total * 100
    ev     = df["return_pct"].mean()
    avg_w  = wins["return_pct"].mean() if len(wins) > 0 else 0
    avg_l  = df[df["return_pct"] <= 0]["return_pct"].mean() if len(df[df["return_pct"] <= 0]) > 0 else 0

    print(f"\n總訊號數：{total} 筆")
    print(f"勝率：{wr:.1f}%（{len(wins)}/{total}）")
    print(f"平均獲利（贏）：{avg_w:+.2f}%")
    print(f"平均虧損（輸）：{avg_l:+.2f}%")
    print(f"期望值（每筆）：{ev:+.2f}%")
    print(f"平均持倉天數：{df['days_held'].mean():.1f} 天")

    print("\n─── 按年份 ───")
    for yr, grp in df.groupby("year"):
        yr_wr = len(grp[grp["return_pct"] > 0]) / len(grp) * 100
        print(f"  {yr}：勝率 {yr_wr:.0f}%，期望值 {grp['return_pct'].mean():+.2f}%，筆數 {len(grp)}")

    print("\n─── 按 entry_score 分組（轉強訊號強度）───")
    for es, grp in df.groupby("entry_score"):
        if len(grp) < 5:
            continue
        yr_wr = len(grp[grp["return_pct"] > 0]) / len(grp) * 100
        print(f"  entry_score={es}：勝率 {yr_wr:.0f}%，期望值 {grp['return_pct'].mean():+.2f}%，筆數 {len(grp)}")

    if df["taiex_green"].notna().any():
        print("\n─── 按大盤三重ST多方條數分組 ───")
        for tg, grp in df.groupby("taiex_green"):
            if len(grp) < 5:
                continue
            yr_wr = len(grp[grp["return_pct"] > 0]) / len(grp) * 100
            print(f"  大盤{int(tg)}綠：勝率 {yr_wr:.0f}%，期望值 {grp['return_pct'].mean():+.2f}%，筆數 {len(grp)}")

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="結構性回調組合回測")
    parser.add_argument("--market", default="tw", choices=["tw", "us"])
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-09-22")
    parser.add_argument("--taiex-min", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--season", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    if args.market == "tw":
        symbols = get_tw_symbols()
        fee_buy, fee_sell = TW_FEE_BUY, TW_FEE_SELL
    else:
        symbols = get_us_symbols()
        fee_buy = fee_sell = 0.0

    print(f"市場：{args.market.upper()}，期間：{args.start} ~ {args.end}，股票數：{len(symbols)}")

    taiex_lookup = {}
    if args.taiex_min > 0:
        taiex_lookup = load_taiex_state(args.start, args.end, args.market)

    if not args.no_download:
        batch_download(symbols, args.start, args.end)

    all_trades = []
    valid_count = 0
    t0 = time.time()
    for i, sym in enumerate(symbols):
        if (i + 1) % 200 == 0:
            print(f"  進度 {i+1}/{len(symbols)}，已用時 {time.time()-t0:.0f}s，訊號累計 {len(all_trades)} 筆")
        df = load_ohlcv(sym, args.start, args.end)
        if df is None or len(df) < 260:
            continue
        valid_count += 1
        trades = simulate_pullback_trades(
            sym, df, fee_buy, fee_sell,
            taiex_lookup=taiex_lookup, taiex_min=args.taiex_min, season_filter=args.season,
        )
        all_trades.extend(trades)

    print_summary(all_trades, args.start, args.end, len(symbols), valid_count)

    if all_trades:
        from datetime import date
        today_str = date.today().strftime("%Y%m%d")
        parts = []
        if args.taiex_min > 0: parts.append(f"taiex{args.taiex_min}")
        if args.season: parts.append("season")
        suffix = ("_" + "_".join(parts)) if parts else ""
        csv_path = RESULT_DIR / f"signals_pullback_{args.market}{suffix}_{today_str}.csv"
        pd.DataFrame(all_trades).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\nCSV 已儲存：{csv_path}")


if __name__ == "__main__":
    main()
