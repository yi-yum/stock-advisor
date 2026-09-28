"""
backtest_portfolio.py — 三重 SuperTrend 組合回測
================================================
進場：三條 ST 全翻多（green_count==3 AND prev_green<3）→ 次日開盤買入
出場：三條 ST 全轉紅（green_count==0）→ 次日開盤賣出
     回測結束日仍持倉 → 最後一日收盤結算

執行：python backtest_portfolio.py
     python backtest_portfolio.py --market tw --start 2021-01-01 --end 2026-09-22
"""

import argparse
import os
import sys
import time
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))
from signal_engine import _supertrend  # 複用現有 ST 計算

# ── 常數 ──────────────────────────────────────────────────────────────────────
ST_PARAMS = [(11, 2.0), (10, 1.0), (12, 3.0)]
CACHE_DIR  = Path("backtest_cache")
RESULT_DIR = Path("backtest_results")
CACHE_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# 台股手續費
TW_FEE_BUY  = 0.001425   # 買 0.1425%
TW_FEE_SELL = 0.001425 + 0.003  # 賣 0.1425% + 0.3% 交易稅

# ── 股票清單 ──────────────────────────────────────────────────────────────────

def get_tw_symbols() -> list[str]:
    """從 scanner.py 的 get_all_tw_symbols() 取得完整台股清單。"""
    from scanner import get_all_tw_symbols
    symbols = get_all_tw_symbols()
    if not symbols:
        print("警告：無法從 API 取得台股清單，使用備援小清單")
        symbols = [
            "2330.TW","2317.TW","2454.TW","2308.TW","2382.TW",
            "2303.TW","2412.TW","2881.TW","2882.TW","1301.TW",
        ]
    return symbols


def get_us_symbols() -> list[str]:
    from scanner import US_SYMBOLS
    return US_SYMBOLS


def get_sp500_symbols() -> list[str]:
    """從 Wikipedia 抓取 S&P500 成分股清單（約 503 支）。"""
    import requests
    from io import StringIO
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=15,
            verify=False,
        )
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        df = tables[0]
        symbols = df["Symbol"].tolist()
        # Wikipedia 用 . 表示 -（如 BRK.B → BRK-B）
        symbols = [s.replace(".", "-") for s in symbols]
        print(f"S&P500 清單取得：{len(symbols)} 支")
        return symbols
    except Exception as e:
        print(f"警告：無法取得 S&P500 清單（{e}），改用 scanner 清單")
        return get_us_symbols()


# ── 資料下載 ──────────────────────────────────────────────────────────────────

def load_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """
    從快取讀取或向 yfinance 下載 OHLCV 日線資料。
    回傳含 Open/High/Low/Close/Volume 的 DataFrame，index 為 date。
    失敗回傳 None。
    """
    cache_file = CACHE_DIR / f"{symbol.replace('.', '_')}.parquet"

    # 嘗試讀取快取
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            # 快取涵蓋所需區間就直接用
            if not df.empty and str(df.index[0].date()) <= start:
                return df[(df.index >= start) & (df.index <= end)]
        except Exception:
            pass  # 快取損壞，重新下載

    # 下載（抓完整歷史備用，start 往前多抓 3 個月暖機期）
    try:
        dl_start = str(pd.Timestamp(start) - pd.DateOffset(months=3))[:10]
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=dl_start, end=end, auto_adjust=True)
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.to_parquet(cache_file)
        return df[(df.index >= start) & (df.index <= end)]
    except Exception as e:
        print(f"  下載失敗 {symbol}: {e}")
        return None


def batch_download(symbols: list[str], start: str, end: str,
                   batch_size: int = 20, delay: float = 1.0):
    """批量下載並快取，顯示進度。已有快取者跳過。"""
    need_download = []
    for sym in symbols:
        cache_file = CACHE_DIR / f"{sym.replace('.', '_')}.parquet"
        if not cache_file.exists():
            need_download.append(sym)

    if not need_download:
        print(f"全部 {len(symbols)} 支均有快取，跳過下載。")
        return

    print(f"需下載 {len(need_download)} 支（共 {len(symbols)} 支）...")
    dl_start = str(pd.Timestamp(start) - pd.DateOffset(months=3))[:10]

    for i in range(0, len(need_download), batch_size):
        batch = need_download[i:i + batch_size]
        print(f"  下載 {i+1}–{min(i+batch_size, len(need_download))}/{len(need_download)}: {batch[:3]}{'...' if len(batch)>3 else ''}")
        try:
            raw = yf.download(
                batch, start=dl_start, end=end,
                auto_adjust=True, group_by="ticker",
                progress=False, threads=True,
            )
            for sym in batch:
                try:
                    if len(batch) == 1:
                        df = raw
                    else:
                        df = raw[sym] if sym in raw.columns.get_level_values(0) else pd.DataFrame()
                    if df is None or df.empty:
                        continue
                    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
                    if df.empty:
                        continue
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    cache_file = CACHE_DIR / f"{sym.replace('.', '_')}.parquet"
                    df.to_parquet(cache_file)
                except Exception:
                    pass
        except Exception as e:
            print(f"  批次下載失敗: {e}")
        if delay > 0:
            time.sleep(delay)

    print("下載完成。")


# ── ST 計算與訊號掃描 ─────────────────────────────────────────────────────────

def calc_st_directions(df: pd.DataFrame) -> np.ndarray:
    """
    回傳 shape (3, N) 的 direction 矩陣，row 0/1/2 對應三組 ST 參數。
    direction: 1=多, -1=空
    """
    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    dirs = []
    for period, mult in ST_PARAMS:
        direction, _ = _supertrend(high, low, close, period, mult)
        dirs.append(direction)
    return np.array(dirs)  # (3, N)


def find_signals(df: pd.DataFrame, dirs: np.ndarray) -> list[dict]:
    """
    掃描進出場訊號。
    回傳 list of dict：
      type  = 'entry' | 'exit'
      date  = 訊號確認日（當天收盤後確認）
      idx   = 在 df 中的 index 位置
    """
    n = df.shape[0]
    green = (dirs == 1).sum(axis=0)  # shape (N,)
    signals = []
    for i in range(1, n):
        cur  = int(green[i])
        prev = int(green[i - 1])
        if cur == 3 and prev < 3:
            signals.append({"type": "entry", "date": df.index[i], "idx": i})
        elif cur == 0 and prev > 0:
            signals.append({"type": "exit",  "date": df.index[i], "idx": i})
    return signals


# ── 回測模擬 ──────────────────────────────────────────────────────────────────

def calc_vol_ratio(df: pd.DataFrame) -> np.ndarray:
    """計算每日量比（當日量 / 20日均量），回傳與 df 等長的陣列。"""
    vol = df["Volume"].values.astype(float)
    vol_ma20 = pd.Series(vol).rolling(20, min_periods=5).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(vol_ma20 > 0, vol / vol_ma20, np.nan)
    return ratio


def calc_signal_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    計算訊號當天可用的特徵（全無未來資訊）。
    回傳 dict of ndarray，每個與 df 等長。
    """
    close  = df["Close"].values.astype(float)
    high   = df["High"].values.astype(float)
    low    = df["Low"].values.astype(float)
    n      = len(close)

    # RSI(14)
    delta  = pd.Series(close).diff()
    gain   = delta.where(delta > 0, 0.0).ewm(span=14, adjust=False).mean().values
    loss   = (-delta.where(delta < 0, 0.0)).ewm(span=14, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100 - 100 / (1 + gain / (loss + 1e-10))

    # ATR(14) / close  →  相對波動率
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - np.roll(close, 1)),
        np.abs(low  - np.roll(close, 1)),
    ])
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        atr_pct = atr / (close + 1e-10)

    # 距 52 週高低點（滾動 252 日）
    roll_high = pd.Series(high).rolling(252, min_periods=20).max().values
    roll_low  = pd.Series(low ).rolling(252, min_periods=20).min().values
    with np.errstate(divide="ignore", invalid="ignore"):
        dist_high = (close - roll_high) / (roll_high + 1e-10)  # 負值，0 = 在高點
        dist_low  = (close - roll_low)  / (roll_low  + 1e-10)  # 正值，大 = 距底遠

    return {
        "rsi":       rsi,
        "atr_pct":   atr_pct,
        "dist_high": dist_high,
        "dist_low":  dist_low,
        "price":     close,
    }


def load_taiex_state(start: str, end: str, market: str = "tw") -> dict:
    """
    下載大盤指數並計算三重 ST 多方條數，回傳 {date_str: green_count} lookup。
    台股用 ^TWII，美股用 ^GSPC。
    """
    ticker = "^TWII" if market == "tw" else "^GSPC"
    label  = "加權指數（^TWII）" if market == "tw" else "S&P 500（^GSPC）"
    print(f"下載{label}...")
    dl_start = str(pd.Timestamp(start) - pd.DateOffset(months=3))[:10]
    try:
        df = yf.Ticker(ticker).history(start=dl_start, end=end, auto_adjust=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[["High", "Low", "Close"]].dropna()
    except Exception as e:
        print(f"警告：無法下載 {ticker}，大盤過濾停用（{e}）")
        return {}

    high  = df["High"].values.astype(float)
    low   = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    dirs  = []
    for period, mult in ST_PARAMS:
        d, _ = _supertrend(high, low, close, period, mult)
        dirs.append(d)
    green = (np.array(dirs) == 1).sum(axis=0)
    lookup = {df.index[i].strftime("%Y-%m-%d"): int(green[i]) for i in range(len(df))}
    print(f"{ticker} 狀態載入完成（{len(lookup)} 天）")
    return lookup


AVOID_MONTHS = {5, 6, 7, 8, 9}  # 5–9月過濾


def simulate_trades(symbol: str, df: pd.DataFrame,
                    fee_buy: float, fee_sell: float,
                    end_date: str,
                    taiex_lookup: dict | None = None,
                    taiex_min: int = 0,
                    season_filter: bool = False) -> list[dict]:
    """
    對單支股票跑完整回測。
    回傳每筆交易 dict。
    """
    if len(df) < 60:
        return []

    dirs = calc_st_directions(df)
    signals = find_signals(df, dirs)
    if not signals:
        return []

    dates     = df.index
    opens     = df["Open"].values.astype(float)
    closes    = df["Close"].values.astype(float)
    vol_ratio = calc_vol_ratio(df)
    feats     = calc_signal_features(df)
    n         = len(df)
    end_ts    = pd.Timestamp(end_date)

    trades = []
    in_trade = False
    entry_date = entry_price = entry_idx = None

    # 建立訊號 lookup：idx → type
    exit_set  = set()
    entry_set = set()
    for s in signals:
        if s["type"] == "entry":
            entry_set.add(s["idx"])
        else:
            exit_set.add(s["idx"])

    signal_vol  = {}  # entry_exec_idx → vol_ratio at signal day
    i = 0
    while i < n:
        if not in_trade:
            if i in entry_set:
                # 次日開盤進場
                exec_idx = i + 1
                if exec_idx >= n:
                    break
                ep = opens[exec_idx]
                if np.isnan(ep) or ep <= 0:
                    i += 1
                    continue
                # 季節過濾（5–9月不進場）
                if season_filter and dates[i].month in AVOID_MONTHS:
                    i += 1
                    continue

                # 大盤過濾
                if taiex_min > 0 and taiex_lookup:
                    sig_date_str = dates[i].strftime("%Y-%m-%d")
                    tg = taiex_lookup.get(sig_date_str)
                    if tg is None:
                        # 找最近一個有資料的交易日
                        tg = next(
                            (taiex_lookup[k] for k in sorted(taiex_lookup.keys(), reverse=True) if k <= sig_date_str),
                            0,
                        )
                    if tg < taiex_min:
                        i += 1
                        continue

                in_trade    = True
                entry_idx   = exec_idx
                entry_date  = dates[exec_idx]
                entry_price = ep * (1 + fee_buy)
                signal_vr   = float(vol_ratio[i]) if not np.isnan(vol_ratio[i]) else np.nan
                signal_feats = {k: (float(v[i]) if not np.isnan(v[i]) else None)
                                for k, v in feats.items()}
                sig_taiex_green = taiex_lookup.get(dates[i].strftime("%Y-%m-%d"), None) if taiex_lookup else None
        else:
            if i in exit_set:
                # 次日開盤出場
                exec_idx = i + 1
                if exec_idx >= n:
                    exec_idx = i
                    xp = closes[exec_idx]
                else:
                    xp = opens[exec_idx]
                if np.isnan(xp) or xp <= 0:
                    i += 1
                    continue
                exit_price = xp * (1 - fee_sell)
                ret = (exit_price - entry_price) / entry_price
                trades.append({
                    "symbol":      symbol,
                    "signal_date": dates[i].date(),
                    "entry_date":  entry_date.date(),
                    "entry_price": round(entry_price, 4),
                    "exit_date":   dates[exec_idx].date(),
                    "exit_price":  round(exit_price, 4),
                    "return_pct":  round(ret * 100, 4),
                    "days_held":   (dates[exec_idx] - entry_date).days,
                    "exit_reason": "3紅出場",
                    "year":        entry_date.year,
                    "vol_ratio":   round(signal_vr, 2) if not np.isnan(signal_vr) else None,
                    "rsi":         round(signal_feats["rsi"],    2) if signal_feats["rsi"]    is not None else None,
                    "atr_pct":     round(signal_feats["atr_pct"],4) if signal_feats["atr_pct"] is not None else None,
                    "dist_high":   round(signal_feats["dist_high"],4) if signal_feats["dist_high"] is not None else None,
                    "dist_low":    round(signal_feats["dist_low"], 4) if signal_feats["dist_low"]  is not None else None,
                    "price":       round(signal_feats["price"],  2) if signal_feats["price"]   is not None else None,
                    "entry_month": entry_date.month,
                    "taiex_green": sig_taiex_green,
                })
                in_trade = False
                entry_date = entry_price = entry_idx = signal_vr = signal_feats = sig_taiex_green = None
        i += 1

    # 回測結束仍持倉 → 最後收盤結算
    if in_trade and entry_price is not None:
        last_idx   = n - 1
        last_close = closes[last_idx]
        last_date  = dates[last_idx]
        if not np.isnan(last_close) and last_close > 0:
            exit_price = last_close * (1 - fee_sell)
            ret = (exit_price - entry_price) / entry_price
            trades.append({
                "symbol":      symbol,
                "signal_date": dates[entry_idx - 1].date() if entry_idx > 0 else entry_date.date(),
                "entry_date":  entry_date.date(),
                "entry_price": round(entry_price, 4),
                "exit_date":   last_date.date(),
                "exit_price":  round(exit_price, 4),
                "return_pct":  round(ret * 100, 4),
                "days_held":   (last_date - entry_date).days,
                "exit_reason": "回測結束",
                "year":        entry_date.year,
                "vol_ratio":   round(signal_vr, 2) if signal_vr and not np.isnan(signal_vr) else None,
                "rsi":         round(signal_feats["rsi"],    2) if signal_feats and signal_feats["rsi"]    is not None else None,
                "atr_pct":     round(signal_feats["atr_pct"],4) if signal_feats and signal_feats["atr_pct"] is not None else None,
                "dist_high":   round(signal_feats["dist_high"],4) if signal_feats and signal_feats["dist_high"] is not None else None,
                "dist_low":    round(signal_feats["dist_low"], 4) if signal_feats and signal_feats["dist_low"]  is not None else None,
                "price":       round(signal_feats["price"],  2) if signal_feats and signal_feats["price"]   is not None else None,
                "entry_month": entry_date.month,
                "taiex_green": sig_taiex_green if signal_feats else None,
            })

    return trades


# ── 統計輸出 ──────────────────────────────────────────────────────────────────

def print_summary(trades: list[dict], start: str, end: str,
                  total_symbols: int, valid_symbols: int):
    df = pd.DataFrame(trades)
    if df.empty:
        print("無任何交易紀錄。")
        return

    wins   = df[df["return_pct"] > 0]
    losses = df[df["return_pct"] <= 0]
    total  = len(df)
    wr     = len(wins) / total * 100
    avg_w  = wins["return_pct"].mean()   if len(wins)   > 0 else 0
    avg_l  = losses["return_pct"].mean() if len(losses) > 0 else 0
    ev     = df["return_pct"].mean()
    avg_d  = df["days_held"].mean()

    print("\n" + "="*50)
    print("  三重 SuperTrend 組合回測結果")
    print("="*50)
    print(f"期間：{start} ~ {end}")
    print(f"股票池：台股（{total_symbols} 支，有效 {valid_symbols} 支）")
    print()
    print(f"總訊號數：{total} 筆")
    print(f"  其中回測結束仍持倉結算：{len(df[df['exit_reason']=='回測結束'])} 筆")
    print()
    print(f"勝率：{wr:.1f}%（{len(wins)}/{total}）")
    print(f"平均獲利（贏）：{avg_w:+.2f}%")
    print(f"平均虧損（輸）：{avg_l:+.2f}%")
    print(f"期望值（每筆）：{ev:+.2f}%")
    print()
    print(f"平均持倉天數：{avg_d:.1f} 天")
    print(f"最長持倉：{df['days_held'].max()} 天")
    print(f"最短持倉：{df['days_held'].min()} 天")

    print()
    print("─── 按年份 ───")
    for yr, grp in df.groupby("year"):
        yr_wins = grp[grp["return_pct"] > 0]
        yr_wr   = len(yr_wins) / len(grp) * 100
        yr_ev   = grp["return_pct"].mean()
        print(f"  {yr}：勝率 {yr_wr:.0f}%，期望值 {yr_ev:+.2f}%，筆數 {len(grp)}")

    print("="*50)


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="三重 SuperTrend 組合回測")
    parser.add_argument("--market",    default="tw", choices=["tw", "us"])
    parser.add_argument("--universe",  default="scanner", choices=["scanner", "sp500"],
                        help="美股股票清單：scanner（預設114支）或 sp500（全S&P500約500支）")
    parser.add_argument("--start",     default="2021-01-01")
    parser.add_argument("--end",       default="2026-09-22")
    parser.add_argument("--taiex-min", type=int, default=0, choices=[0, 1, 2, 3],
                        help="大盤過濾：訊號日加權指數三重ST至少幾條多方才進場（0=不過濾）")
    parser.add_argument("--season", action="store_true",
                        help="季節過濾：5–9月不進場，只做10–4月")
    parser.add_argument("--no-download", action="store_true",
                        help="跳過批量下載，只用現有快取")
    args = parser.parse_args()

    index_label = "加權指數" if args.market == "tw" else "S&P500"
    parts = []
    if args.taiex_min > 0: parts.append(f"{index_label}>={args.taiex_min}綠")
    if args.season:         parts.append("只做10–4月")
    filter_label = " + ".join(parts) if parts else "無過濾"
    print(f"市場：{args.market.upper()}，期間：{args.start} ~ {args.end}，過濾：{filter_label}")

    # 取得股票清單
    if args.market == "tw":
        symbols  = get_tw_symbols()
        fee_buy  = TW_FEE_BUY
        fee_sell = TW_FEE_SELL
    else:
        symbols  = get_sp500_symbols() if args.universe == "sp500" else get_us_symbols()
        fee_buy  = fee_sell = 0.0

    print(f"股票清單：{len(symbols)} 支")

    # 大盤狀態載入
    taiex_lookup = {}
    if args.taiex_min > 0:
        taiex_lookup = load_taiex_state(args.start, args.end, args.market)

    # 批量下載（首次耗時較長）
    if not args.no_download:
        batch_download(symbols, args.start, args.end)

    # 逐股回測
    all_trades   = []
    valid_count  = 0
    t0 = time.time()

    for i, sym in enumerate(symbols):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  進度 {i+1}/{len(symbols)}，已用時 {elapsed:.0f}s，"
                  f"訊號累計 {len(all_trades)} 筆")

        df = load_ohlcv(sym, args.start, args.end)
        if df is None or len(df) < 60:
            continue
        valid_count += 1

        trades = simulate_trades(sym, df, fee_buy, fee_sell, args.end,
                                 taiex_lookup=taiex_lookup,
                                 taiex_min=args.taiex_min,
                                 season_filter=args.season)
        all_trades.extend(trades)

    # 輸出統計
    print_summary(all_trades, args.start, args.end, len(symbols), valid_count)

    # 儲存 CSV
    if all_trades:
        today_str = date.today().strftime("%Y%m%d")
        index_suffix = "taiex" if args.market == "tw" else "gspc"
        parts_s = []
        if args.market == "us" and args.universe == "sp500": parts_s.append("sp500")
        if args.taiex_min > 0: parts_s.append(f"{index_suffix}{args.taiex_min}")
        if args.season:         parts_s.append("season")
        suffix = ("_" + "_".join(parts_s)) if parts_s else ""
        csv_path  = RESULT_DIR / f"signals_{args.market}{suffix}_{today_str}.csv"
        pd.DataFrame(all_trades).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\nCSV 已儲存：{csv_path}")


if __name__ == "__main__":
    main()
