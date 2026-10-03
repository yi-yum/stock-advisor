"""
crypto_tracker.py — 加密貨幣訊號持倉追蹤

由 router_crypto.check_and_notify()（每小時）呼叫 update_tracker()：
  · 新進場訊號（LONG/SHORT）→ 開倉，記錄進場價、止損、止盈、加碼點位
  · 策略出場訊號（EXIT）/ 觸及止損 / 觸及止盈 → 平倉並記錄損益
  · 其餘時間更新現價與未實現損益

目的：讓每套幣種策略都有「即時運行後的真實紀錄」可以驗證，
而不是只靠回測。損益已扣除往返手續費（沿用 backtest_crypto.FEE，單邊 0.05%）。

限制：止損/止盈只在每小時檢查時用當下價格判斷，小時內的插針不會被捕捉；
後端沒開著的時段錯過的進出場訊號不會回補。

資料檔：crypto_tracker.json
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from json_utils import atomic_write_json
from signal_engine import calc_add_on_levels

TRACKER_FILE = Path(__file__).parent / "crypto_tracker.json"
_LOCK = threading.Lock()
MAX_CLOSED = 500


def _fee() -> float:
    from backtest_crypto import FEE
    return FEE


def _load() -> dict:
    if TRACKER_FILE.exists():
        try:
            data = json.loads(TRACKER_FILE.read_text(encoding="utf-8"))
            data.setdefault("open", [])
            data.setdefault("closed", [])
            return data
        except Exception:
            pass
    return {"open": [], "closed": []}


def _save(data: dict) -> None:
    data["closed"] = data["closed"][-MAX_CLOSED:]
    atomic_write_json(TRACKER_FILE, data, ensure_ascii=False, indent=2)


def _pnl_pct(side: str, entry: float, price: float) -> float:
    raw = (price - entry) / entry if side == "LONG" else (entry - price) / entry
    return round((raw - _fee() * 2) * 100, 2)


def _stop_price(cfg: dict, side: str, entry: float, status: dict) -> Optional[float]:
    """優先用策略自己算出的 SL（VB），否則用回測設定的固定停損比例。"""
    sl = status.get("sl")
    if sl is not None:
        return float(sl)
    pct = cfg.get("stop_pct") or cfg.get("stop_loss")
    if not pct:
        return None
    return round(entry * (1 - pct) if side == "LONG" else entry * (1 + pct), 4)


def _hours_between(start_iso: str, end_iso: str) -> float:
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    try:
        a = datetime.strptime(start_iso, fmt)
        b = datetime.strptime(end_iso, fmt)
        return round((b - a).total_seconds() / 3600, 1)
    except Exception:
        return 0.0


def _close(pos: dict, price: float, now_str: str, reason: str) -> dict:
    return {
        "symbol":       pos["symbol"],
        "strategy":     pos["strategy"],
        "side":         pos["side"],
        "entry_time":   pos["entry_time"],
        "entry_price":  pos["entry_price"],
        "exit_time":    now_str,
        "exit_price":   price,
        "pnl_pct":      _pnl_pct(pos["side"], pos["entry_price"], price),
        "exit_reason":  reason,
        "holding_hours": _hours_between(pos["entry_time"], now_str),
        "entry_reason": pos.get("entry_reason", ""),
        "sl":           pos.get("sl"),
        "tp":           pos.get("tp"),
    }


def _hit_stop(pos: dict, price: float) -> bool:
    sl = pos.get("sl")
    if sl is None:
        return False
    return price <= sl if pos["side"] == "LONG" else price >= sl


def _hit_target(pos: dict, price: float) -> bool:
    tp = pos.get("tp")
    if tp is None:
        return False
    return price >= tp if pos["side"] == "LONG" else price <= tp


def update_tracker(alerts: list, current_map: dict, now_str: str, coin_cfg: dict) -> dict:
    """
    alerts:      check_and_notify 判定的「新」進出場事件（signal 為 LONG/SHORT/EXIT）
    current_map: {symbol: 目前策略狀態 dict（含 price/sl/tp/reason）}
    coin_cfg:    backtest_crypto.COIN_BEST（用來取固定停損比例）
    回傳本次的變動摘要。
    """
    with _LOCK:
        data = _load()
        exit_syms = {a["symbol"] for a in alerts if a.get("signal") == "EXIT"}
        opened, closed_now = [], []

        still_open = []
        for pos in data["open"]:
            cur = current_map.get(pos["symbol"])
            price = cur.get("price") if cur else None
            if price is None:
                still_open.append(pos)
                continue

            reason = None
            if pos["symbol"] in exit_syms:
                reason = "策略出場訊號"
            elif _hit_stop(pos, price):
                reason = "觸及止損"
            elif _hit_target(pos, price):
                reason = "觸及止盈"

            if reason:
                rec = _close(pos, price, now_str, reason)
                data["closed"].append(rec)
                closed_now.append({"symbol": rec["symbol"], "pnl_pct": rec["pnl_pct"], "reason": reason})
                continue

            pos["current_price"]   = price
            pos["current_pnl_pct"] = _pnl_pct(pos["side"], pos["entry_price"], price)
            pos["last_updated"]    = now_str
            still_open.append(pos)

        tracked = {p["symbol"] for p in still_open}
        for a in alerts:
            sig, sym = a.get("signal"), a.get("symbol")
            if sig not in ("LONG", "SHORT") or sym in tracked:
                continue
            price = a.get("price")
            if price is None:
                continue
            cfg = coin_cfg.get(sym, {})
            sl = _stop_price(cfg, sig, price, a)
            pos = {
                "symbol":          sym,
                "strategy":        a.get("strategy", ""),
                "side":            sig,
                "entry_time":      now_str,
                "entry_price":     price,
                "entry_reason":    a.get("reason", ""),
                "sl":              sl,
                "tp":              a.get("tp"),
                "add_on_levels":   calc_add_on_levels(price, sl) if sig == "LONG" else [],
                "current_price":   price,
                "current_pnl_pct": _pnl_pct(sig, price, price),
                "last_updated":    now_str,
            }
            still_open.append(pos)
            tracked.add(sym)
            opened.append(sym)

        data["open"] = still_open
        _save(data)
        return {"opened": opened, "closed": closed_now}


def get_tracker(closed_limit: int = 100) -> dict:
    data = _load()
    return {
        "open":    sorted(data["open"], key=lambda p: p["entry_time"], reverse=True),
        "closed":  sorted(data["closed"], key=lambda p: p["exit_time"], reverse=True)[:closed_limit],
        "summary": _summary(data),
    }


def _stats(rows: list) -> dict:
    n = len(rows)
    if n == 0:
        return {"trades": 0, "win_rate": None, "avg_pnl": None, "total_pnl": None, "avg_hours": None}
    wins = sum(1 for r in rows if r["pnl_pct"] > 0)
    return {
        "trades":    n,
        "win_rate":  round(wins / n * 100, 1),
        "avg_pnl":   round(sum(r["pnl_pct"] for r in rows) / n, 2),
        "total_pnl": round(sum(r["pnl_pct"] for r in rows), 2),
        "avg_hours": round(sum(r["holding_hours"] for r in rows) / n, 1),
    }


def _summary(data: dict) -> dict:
    closed = data["closed"]
    open_ = data["open"]
    by_strategy, by_symbol = {}, {}
    for r in closed:
        by_strategy.setdefault(r["strategy"], []).append(r)
        by_symbol.setdefault(r["symbol"], []).append(r)
    unrealized = [p["current_pnl_pct"] for p in open_ if p.get("current_pnl_pct") is not None]
    return {
        "overall":        _stats(closed),
        "by_strategy":    {k: _stats(v) for k, v in by_strategy.items()},
        "by_symbol":      {k: _stats(v) for k, v in by_symbol.items()},
        "open_count":     len(open_),
        "open_unrealized_avg": round(sum(unrealized) / len(unrealized), 2) if unrealized else None,
    }
