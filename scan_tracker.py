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

import yfinance as yf

logger = logging.getLogger("scan_tracker")

TRACKER_FILE = Path(__file__).parent / "scan_tracker.json"


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
    with open(TRACKER_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── OHLC 快照 ─────────────────────────────────────────────────────────────────

def _fetch_entry_day_ohlc(symbol: str, market: str) -> Optional[dict]:
    """抓取最近一個交易日的 OHLC，作為進場當天快照"""
    try:
        ticker = symbol if market == "us" else symbol  # TW 符號已含 .TW/.TWO
        df = yf.download(ticker, period="3d", auto_adjust=True, progress=False)
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

        sym        = pos["symbol"]
        entry_price = pos["entry_price"]
        sl          = pos.get("sl")

        if sym in today_map:
            # 仍在掃描結果中
            current_result = today_map[sym]
            current_price  = current_result.get("close", entry_price)
            pnl_pct        = round((current_price - entry_price) / entry_price * 100, 2)

            # 檢查是否跌破止損
            if sl and current_price < sl:
                closed_positions.append(_close_position(
                    pos, current_price, today, "跌破止損"
                ))
                logger.info(f"[Tracker] {sym} 跌破止損 {sl}，收盤 {current_price}，PnL {pnl_pct:.1f}%")
            else:
                # 更新現況
                pos["current_price"]  = current_price
                pos["current_pnl_pct"] = pnl_pct
                pos["last_updated"]   = today
                still_open.append(pos)
        else:
            # 不在今日掃描結果 → 訊號消失，關倉
            # 用上次已知收盤價（沒有今日資料就用 last_known）
            exit_price = pos.get("current_price", entry_price)
            pnl_pct    = round((exit_price - entry_price) / entry_price * 100, 2)
            closed_positions.append(_close_position(
                pos, exit_price, today, "訊號消失"
            ))
            logger.info(f"[Tracker] {sym} 訊號消失，PnL {pnl_pct:.1f}%")

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
        "symbol":       pos["symbol"],
        "market":       pos.get("market", ""),
        "sector":       pos.get("sector", ""),
        "signal":       pos.get("signal"),
        "signal_label": pos.get("signal_label"),
        "strategy":     pos.get("strategy", ""),
        "entry_date":   pos["entry_date"],
        "entry_price":  entry_price,
        "exit_date":    exit_date,
        "exit_price":   exit_price,
        "pnl_pct":      pnl_pct,
        "exit_reason":  reason,
        "holding_days": holding_days,
        "trend_score":  pos.get("trend_score"),
        "entry_score":  pos.get("entry_score"),
        "entry_timing": pos.get("entry_timing"),
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
