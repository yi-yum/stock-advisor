"""
scan_tracker.py — 掃描訊號自動追蹤器

每次每日掃描完成後呼叫 update_tracker()：
  1. BUY 訊號 → 若尚未追蹤，加入開放部位
  2. WATCH 訊號 → 已開倉的繼續持有；新的不自動開倉（避免雜訊）
  3. 下次掃描若不再出現，或收盤跌破止損 → 自動關倉，記錄損益

資料檔：scan_tracker.json
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yfinance as yf

from json_utils import atomic_write_json

logger = logging.getLogger("scan_tracker")

TRACKER_FILE      = Path(__file__).parent / "scan_tracker.json"
INTRADAY_SNAP_DIR = Path(__file__).parent / "intraday_snapshots"
INTRADAY_SNAP_DIR.mkdir(exist_ok=True)


# ── 讀寫 ──────────────────────────────────────────────────────────────────────

def _load() -> dict:
    if TRACKER_FILE.exists():
        try:
            with open(TRACKER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"open": [], "closed": []}


def _save(data: dict):
    atomic_write_json(TRACKER_FILE, data, ensure_ascii=False, indent=2)


# ── OHLC 快照 ─────────────────────────────────────────────────────────────────

def _fetch_entry_day_ohlc(symbol: str, market: str) -> Optional[dict]:
    """抓取最近一個交易日的 OHLC，作為進場當天快照"""
    try:
        ticker = symbol if market == "us" else symbol  # TW 符號已含 .TW/.TWO
        df = yf.Ticker(ticker).history(period="3d", auto_adjust=True)
        if df.empty:
            return None
        row = df.iloc[-1]
        return {
            "date":   df.index[-1].strftime("%Y-%m-%d"),
            "open":   round(float(row["Open"]),  2),
            "high":   round(float(row["High"]),  2),
            "low":    round(float(row["Low"]),   2),
            "close":  round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
            "green":  float(row["Close"]) > float(row["Open"]),  # 收紅？
        }
    except Exception as e:
        logger.warning(f"[Tracker] 取得 {symbol} OHLC 失敗: {e}")
        return None


def _save_intraday_snapshot(symbol: str, market: str, date_str: str):
    """
    抓取並儲存當日 5m / 15m / 30m K 棒到 intraday_snapshots/{date}_{market}_{symbol}.json。
    yfinance 限制：5m/15m/30m 只能回溯 60 天，需在訊號當天或隔天立即呼叫。
    """
    safe_sym  = symbol.replace(".", "_").replace("/", "_")
    snap_file = INTRADAY_SNAP_DIR / f"{date_str}_{market}_{safe_sym}.json"
    if snap_file.exists():
        return  # 已存在，不重複抓

    snapshot = {"symbol": symbol, "market": market, "date": date_str, "timeframes": {}}

    for interval in ("5m", "15m", "30m"):
        try:
            df = yf.Ticker(symbol).history(period="2d", interval=interval, auto_adjust=True)
            if df.empty:
                continue
            # 只保留 date_str 當天的資料
            df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index
            day_df = df[df.index.strftime("%Y-%m-%d") == date_str]
            if day_df.empty:
                # fallback：取最後一個交易日
                last_date = df.index[-1].strftime("%Y-%m-%d")
                day_df = df[df.index.strftime("%Y-%m-%d") == last_date]

            bars = []
            for ts, row in day_df.iterrows():
                bars.append({
                    "time":   ts.strftime("%H:%M"),
                    "open":   round(float(row["Open"]),  4),
                    "high":   round(float(row["High"]),  4),
                    "low":    round(float(row["Low"]),   4),
                    "close":  round(float(row["Close"]), 4),
                    "volume": int(row["Volume"]),
                })
            snapshot["timeframes"][interval] = bars
            logger.info(f"[Tracker] {symbol} {interval} 快照：{len(bars)} 根")
        except Exception as e:
            logger.warning(f"[Tracker] {symbol} {interval} 抓取失敗: {e}")

    atomic_write_json(snap_file, snapshot, ensure_ascii=False, indent=2)
    logger.info(f"[Tracker] 日內快照已存：{snap_file.name}")


# ── ST 狀態即時查詢 ───────────────────────────────────────────────────────────

def _get_current_st_state(symbol: str) -> Optional[tuple]:
    """
    用 yfinance 取得最新日線，計算三重 ST 的 (green_count, prev_green, close)。
    失敗時回傳 None（不因資料問題強制出場）。
    """
    try:
        from signal_engine import _supertrend, _ST_PARAMS
        df = yf.Ticker(symbol).history(period="120d", auto_adjust=True)
        if df.empty or len(df) < 60:
            logger.warning(f"[Tracker] {symbol} 資料不足，無法計算 ST 狀態")
            return None
        high  = df["High"].values.astype(float)
        low   = df["Low"].values.astype(float)
        close = df["Close"].values.astype(float)
        st_results = [_supertrend(high, low, close, p, m) for p, m in _ST_PARAMS]
        dirs  = [r[0] for r in st_results]
        lines = [r[1] for r in st_results]
        cur_dirs  = [int(d[-1]) for d in dirs]
        prev_dirs = [int(d[-2]) for d in dirs]
        green_count = sum(1 for d in cur_dirs  if d == 1)
        prev_green  = sum(1 for d in prev_dirs if d == 1)
        # 當日止損：三條 ST 支撐線最高值（最近的防守位）
        current_sl  = round(max(float(lines[i][-1]) for i in range(3)), 4)
        return green_count, prev_green, float(close[-1]), current_sl
    except Exception as e:
        logger.warning(f"[Tracker] 無法取得 {symbol} ST 狀態: {e}")
        return None


# ── 核心更新邏輯 ───────────────────────────────────────────────────────────────

def update_tracker(scan_output: dict):
    """
    每次掃描結束後呼叫。
    scan_output 為 run_tw_scan / run_us_scan 的回傳值。

    掃描結果格式（每個 result）：
      symbol, close, signal, signal_label, strategy,
      sl, tp, rr, trend_score, entry_score, pullback_valid,
      sector, gemini_analysis
    """
    market     = scan_output.get("market", "")
    today      = datetime.now().strftime("%Y-%m-%d")
    results    = scan_output.get("results", [])

    # 今日掃描：建立 symbol → result 的 lookup（只保留 BUY/WATCH）
    today_map: dict[str, dict] = {
        r["symbol"]: r
        for r in results
        if r.get("signal") in ("BUY", "WATCH")
    }
    today_buy_set = {s for s, r in today_map.items() if r.get("signal") == "BUY"}

    data = _load()
    open_positions  = data["open"]
    closed_positions = data["closed"]

    # ── Step 1：更新既有開放部位 ──────────────────────────────
    still_open = []
    for pos in open_positions:
        if pos.get("market") != market:
            still_open.append(pos)   # 不同市場不處理
            continue

        sym         = pos["symbol"]
        entry_price = pos["entry_price"]
        stored_green = pos.get("green_count", 3)  # 上次記錄的 green_count

        # 即時計算目前 ST 狀態
        st_state = _get_current_st_state(sym)
        if st_state is None:
            # 資料取得失敗：保留部位，不強制出場
            logger.warning(f"[Tracker] {sym} ST 狀態取得失敗，保留部位")
            still_open.append(pos)
            continue

        curr_green, _prev_green, current_price, current_sl = st_state
        pnl_pct = round((current_price - entry_price) / entry_price * 100, 2)

        # ── 出場規則（三重ST翻空邏輯）──────────────────────────
        # 規則 1：三條全翻空 → 無條件出場
        if curr_green == 0:
            closed_positions.append(_close_position(
                pos, current_price, today, "三條ST全翻空"
            ))
            logger.info(f"[Tracker] {sym} 三條ST全翻空，PnL {pnl_pct:.1f}%")
            continue

        # 規則 2：ST 條數減少 且 收盤跌破進場價 → 出場
        if curr_green < stored_green and current_price < entry_price:
            flipped = stored_green - curr_green
            reason = f"第{flipped}條ST翻空，跌破進場價"
            closed_positions.append(_close_position(
                pos, current_price, today, reason
            ))
            logger.info(f"[Tracker] {sym} {reason}，PnL {pnl_pct:.1f}%")
            continue

        # 繼續持倉：更新最新狀態
        pos["green_count"]     = curr_green
        pos["current_sl"]      = current_sl   # 當日動態止損
        pos["current_price"]   = current_price
        pos["current_pnl_pct"] = pnl_pct
        pos["last_updated"]    = today
        still_open.append(pos)
        if curr_green < stored_green:
            logger.info(f"[Tracker] {sym} ST 減少至 {curr_green}/3，但收盤高於進場價（{current_price:.2f} > {entry_price:.2f}），持倉繼續")

    # ── Step 2：加入新的 BUY 部位 ────────────────────────────
    tracked_symbols = {p["symbol"] for p in still_open}

    for sym in today_buy_set:
        if sym in tracked_symbols:
            continue   # 已在追蹤中

        r     = today_map[sym]
        entry = r.get("close", 0)

        new_pos = {
            "symbol":          sym,
            "market":          market,
            "sector":          r.get("sector", ""),
            "entry_date":      today,
            "entry_price":     entry,
            "signal":          r.get("signal"),
            "signal_label":    r.get("signal_label"),
            "strategy":        r.get("strategy", ""),
            "trend_score":     r.get("trend_score", 0),
            "entry_score":     r.get("entry_score", 0),
            "pullback_valid":  r.get("pullback_valid", False),
            "sl":              r.get("sl"),
            "tp":              r.get("tp"),
            "rr":              r.get("rr"),
            "add_on_levels":   r.get("add_on_levels", []),
            # ST 翻空出場追蹤
            "green_count":     3,
            "current_sl":      r.get("sl"),  # 進場當天先用 signal_engine 的 sl，後續每日更新
            # 追蹤狀態
            "current_price":   entry,
            "current_pnl_pct": 0.0,
            "last_updated":    today,
            # 進場時機（Claude 分析後補填）
            "entry_timing":    None,
            # 進場當天 OHLC 快照（用於事後統計開盤確認過濾效果）
            "entry_day_ohlc":  _fetch_entry_day_ohlc(sym, market),
        }
        still_open.append(new_pos)
        logger.info(f"[Tracker] 新增追蹤 {sym}（{r.get('signal_label')}），進場價 {entry}")

        # 儲存日內 K 棒快照（5m/15m/30m），供事後驗證開盤確認過濾效果
        _save_intraday_snapshot(sym, market, today)

    data["open"]   = still_open
    data["closed"] = closed_positions
    _save(data)
    logger.info(
        f"[Tracker] {market.upper()} 更新完成：開放 {len(still_open)} 筆，"
        f"歷史 {len(closed_positions)} 筆"
    )


def _close_position(pos: dict, exit_price: float, exit_date: str, reason: str) -> dict:
    entry_price  = pos["entry_price"]
    pnl_pct      = round((exit_price - entry_price) / entry_price * 100, 2)
    entry_dt     = datetime.strptime(pos["entry_date"], "%Y-%m-%d")
    exit_dt      = datetime.strptime(exit_date, "%Y-%m-%d")
    holding_days = (exit_dt - entry_dt).days

    return {
        "symbol":         pos["symbol"],
        "market":         pos.get("market", ""),
        "sector":         pos.get("sector", ""),
        "signal":         pos.get("signal"),
        "signal_label":   pos.get("signal_label"),
        "strategy":       pos.get("strategy", ""),
        "entry_date":     pos["entry_date"],
        "entry_price":    entry_price,
        "exit_date":      exit_date,
        "exit_price":     exit_price,
        "pnl_pct":        pnl_pct,
        "exit_reason":    reason,
        "holding_days":   holding_days,
        "trend_score":    pos.get("trend_score"),
        "entry_score":    pos.get("entry_score"),
        "entry_timing":   pos.get("entry_timing"),
        "claude_analysis": pos.get("claude_analysis"),
        "sl":              pos.get("sl"),
        "tp":              pos.get("tp"),
        "rr":              pos.get("rr"),
        "entry_day_ohlc":  pos.get("entry_day_ohlc"),
    }


def update_entry_timing(market: str, timing_map: dict):
    """
    Claude 分析完成後呼叫，補填開放部位的 entry_timing。
    timing_map: {symbol: "🟢" | "🟡" | "🔴"}
    """
    data = _load()
    updated = 0
    for pos in data["open"]:
        if pos.get("market") != market:
            continue
        sym = pos["symbol"]
        if sym in timing_map and timing_map[sym]:
            pos["entry_timing"] = timing_map[sym]
            updated += 1
    if updated:
        _save(data)
        logger.info(f"[Tracker] 補填 {market.upper()} 進場時機標籤 {updated} 筆")


def update_entry_analysis(market: str, analysis_map: dict):
    """
    Claude 分析完成後呼叫，補填開放部位的完整分析文字。
    analysis_map: {symbol: "分析文字"}
    """
    data = _load()
    updated = 0
    for pos in data["open"]:
        if pos.get("market") != market:
            continue
        sym = pos["symbol"]
        if sym in analysis_map and analysis_map[sym]:
            pos["claude_analysis"] = analysis_map[sym]
            updated += 1
    if updated:
        _save(data)
        logger.info(f"[Tracker] 補填 {market.upper()} Claude 分析文字 {updated} 筆")


# ── 查詢介面 ───────────────────────────────────────────────────────────────────

def get_open_positions(market: Optional[str] = None) -> list:
    data = _load()
    positions = data.get("open", [])
    if market:
        positions = [p for p in positions if p.get("market") == market]
    return sorted(positions, key=lambda p: p.get("entry_date", ""), reverse=True)


def get_closed_positions(market: Optional[str] = None, limit: int = 100) -> list:
    data = _load()
    positions = data.get("closed", [])
    if market:
        positions = [p for p in positions if p.get("market") == market]
    return sorted(positions, key=lambda p: p.get("exit_date", ""), reverse=True)[:limit]


def get_summary() -> dict:
    """取得整體績效摘要"""
    data    = _load()
    closed  = data.get("closed", [])
    open_p  = data.get("open", [])

    if not closed:
        return {
            "total_closed":   0,
            "win_rate":       None,
            "avg_pnl_pct":    None,
            "total_pnl_pct":  None,
            "best_trade":     None,
            "worst_trade":    None,
            "avg_hold_days":  None,
            "open_count":     len(open_p),
            "open_unrealized_pct": None,
        }

    pnl_list    = [p["pnl_pct"] for p in closed]
    wins        = [p for p in pnl_list if p > 0]
    open_pnl    = [p.get("current_pnl_pct", 0) for p in open_p if p.get("current_pnl_pct") is not None]

    best  = max(closed, key=lambda p: p["pnl_pct"])
    worst = min(closed, key=lambda p: p["pnl_pct"])

    # ── 按進場時機分組統計 ──────────────────────────────────────
    timing_stats: dict[str, dict] = {}
    for p in closed:
        t = p.get("entry_timing")
        if not t:
            continue
        if t not in timing_stats:
            timing_stats[t] = {"count": 0, "wins": 0, "total_pnl": 0.0}
        timing_stats[t]["count"] += 1
        if p["pnl_pct"] > 0:
            timing_stats[t]["wins"] += 1
        timing_stats[t]["total_pnl"] += p["pnl_pct"]
    for t, s in timing_stats.items():
        s["win_rate"] = round(s["wins"] / s["count"] * 100, 1) if s["count"] else None
        s["avg_pnl"]  = round(s["total_pnl"] / s["count"], 2) if s["count"] else None

    # ── 按板塊分組統計 ──────────────────────────────────────────
    sector_stats: dict[str, dict] = {}
    for p in closed:
        sec = p.get("sector") or "未分類"
        if sec not in sector_stats:
            sector_stats[sec] = {"count": 0, "wins": 0, "total_pnl": 0.0}
        sector_stats[sec]["count"] += 1
        if p["pnl_pct"] > 0:
            sector_stats[sec]["wins"] += 1
        sector_stats[sec]["total_pnl"] += p["pnl_pct"]
    for sec, s in sector_stats.items():
        s["win_rate"] = round(s["wins"] / s["count"] * 100, 1) if s["count"] else None
        s["avg_pnl"]  = round(s["total_pnl"] / s["count"], 2) if s["count"] else None
    sector_list = sorted(sector_stats.items(), key=lambda x: x[1]["count"], reverse=True)

    # ── 按月份統計損益 ──────────────────────────────────────────
    monthly: dict[str, dict] = {}
    for p in closed:
        month = p.get("exit_date", "")[:7]  # YYYY-MM
        if not month:
            continue
        if month not in monthly:
            monthly[month] = {"count": 0, "wins": 0, "total_pnl": 0.0}
        monthly[month]["count"] += 1
        if p["pnl_pct"] > 0:
            monthly[month]["wins"] += 1
        monthly[month]["total_pnl"] += p["pnl_pct"]
    for m, s in monthly.items():
        s["win_rate"] = round(s["wins"] / s["count"] * 100, 1) if s["count"] else None
        s["avg_pnl"]  = round(s["total_pnl"] / s["count"], 2) if s["count"] else None
        s["total_pnl"] = round(s["total_pnl"], 2)
    monthly_list = sorted(monthly.items())

    return {
        "total_closed":   len(closed),
        "win_rate":       round(len(wins) / len(closed) * 100, 1),
        "avg_pnl_pct":    round(sum(pnl_list) / len(pnl_list), 2),
        "total_pnl_pct":  round(sum(pnl_list), 2),
        "best_trade":     {"symbol": best["symbol"], "pnl_pct": best["pnl_pct"], "exit_date": best["exit_date"]},
        "worst_trade":    {"symbol": worst["symbol"], "pnl_pct": worst["pnl_pct"], "exit_date": worst["exit_date"]},
        "avg_hold_days":  round(sum(p.get("holding_days", 0) for p in closed) / len(closed), 1),
        "open_count":     len(open_p),
        "open_unrealized_pct": round(sum(open_pnl) / len(open_pnl), 2) if open_pnl else None,
        # 新增分析維度
        "timing_stats":   timing_stats,
        "sector_stats":   [{"sector": s, **v} for s, v in sector_list],
        "monthly_stats":  [{"month": m, **v} for m, v in monthly_list],
    }
