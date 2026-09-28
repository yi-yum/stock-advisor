import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timezone
from watchlist import load_watchlist, add_to_watchlist, remove_from_watchlist
from stock_data import get_stock_data
from signal_engine import get_signal, calc_indicators

router = APIRouter(prefix="/api/watchlist", tags=["watchlist"])


class AddWatchlistRequest(BaseModel):
    symbol: str
    note: str = ""


class ScanWatchlistRequest(BaseModel):
    symbols: List[str]


@router.get("")
def get_watchlist():
    """Get all watchlist items"""
    try:
        watchlist = load_watchlist()
        return watchlist
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("")
def add_watchlist(request: AddWatchlistRequest):
    """Add a symbol to watchlist"""
    try:
        success, message = add_to_watchlist(request.symbol, request.note)
        return {"success": success, "message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{symbol}")
def delete_watchlist(symbol: str):
    """Remove a symbol from watchlist"""
    try:
        success, message = remove_from_watchlist(symbol)
        return {"success": success, "message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scan")
def scan_watchlist(request: ScanWatchlistRequest):
    """Scan multiple symbols and return analysis results"""
    try:
        results = []

        for symbol in request.symbols:
            try:
                stock_data, _, error = get_stock_data(symbol)

                if error:
                    results.append({
                        "symbol": symbol,
                        "asset_type": None,
                        "current_price": None,
                        "change_pct": None,
                        "rsi": None,
                        "macd_histogram": None,
                        "pct_from_52w_high": None,
                        "bias_weekly": None,
                        "bias_daily": None,
                        "error": error
                    })
                else:
                    # Extract timeframes data
                    timeframes = stock_data.get("timeframes", {})
                    weekly_bias = timeframes.get("weekly", {}).get("bias") if timeframes.get("weekly") else None
                    daily_bias = timeframes.get("daily", {}).get("bias") if timeframes.get("daily") else None

                    results.append({
                        "symbol": symbol,
                        "asset_type": stock_data.get("asset_type"),
                        "current_price": stock_data.get("current_price"),
                        "change_pct": stock_data.get("change_pct"),
                        "rsi": stock_data.get("rsi"),
                        "macd_histogram": stock_data.get("macd_histogram"),
                        "pct_from_52w_high": stock_data.get("pct_from_52w_high"),
                        "bias_weekly": weekly_bias,
                        "bias_daily": daily_bias,
                        "error": None
                    })
            except Exception as symbol_error:
                results.append({
                    "symbol": symbol,
                    "asset_type": None,
                    "current_price": None,
                    "change_pct": None,
                    "rsi": None,
                    "macd_histogram": None,
                    "pct_from_52w_high": None,
                    "bias_weekly": None,
                    "bias_daily": None,
                    "error": str(symbol_error)
                })

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals")
def get_watchlist_signals():
    """取得自選股的技術入場訊號（一次掃描全部）"""
    try:
        watchlist = load_watchlist()
        results = []
        updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        for item in watchlist:
            symbol = item["symbol"]
            try:
                stock_data, df_raw, error = get_stock_data(symbol)
                if error or stock_data is None:
                    results.append({
                        "symbol": symbol,
                        "asset_type": item.get("asset_type"),
                        "note": item.get("note", ""),
                        "error": error or "無法取得資料",
                        "updated_at": updated_at,
                    })
                    continue

                if df_raw is not None and not df_raw.empty:
                    entry = get_signal(df_raw)
                else:
                    entry = {
                        "regime": "UNKNOWN", "strategy": "", "signal": "WAIT",
                        "signal_label": "資料不足", "can_enter": False, "score": 0,
                        "reasons": [], "entry_price": None, "sl": None, "tp": None, "rr": None,
                        "trigger_reasons": [], "indicators": {},
                    }
                results.append({
                    "symbol": symbol,
                    "asset_type": stock_data.get("asset_type"),
                    "note": item.get("note", ""),
                    "price": stock_data.get("current_price"),
                    "change_pct": stock_data.get("change_pct"),
                    "rsi": stock_data.get("rsi"),
                    "macd_histogram": stock_data.get("macd_histogram"),
                    "volume_ratio": stock_data.get("volume_ratio"),
                    "above_ma20": stock_data.get("above_ma20"),
                    "ma20": stock_data.get("ma20"),
                    "ma20_slope": stock_data.get("ma20_slope"),
                    "pct_from_52w_high": stock_data.get("pct_from_52w_high"),
                    "bb_position": stock_data.get("bb_position"),
                    "regime": entry["regime"],
                    "strategy": entry["strategy"],
                    "signal": entry["signal"],
                    "signal_label": entry["signal_label"],
                    "can_enter": entry["can_enter"],
                    "score": entry["score"],
                    "reasons": entry["reasons"],
                    "entry_price": entry.get("entry_price"),
                    "trigger_reasons": entry.get("trigger_reasons", []),
                    "sl": entry["sl"],
                    "tp": entry["tp"],
                    "rr": entry["rr"],
                    "updated_at": updated_at,
                    "error": None,
                })
            except Exception as e:
                results.append({
                    "symbol": symbol,
                    "asset_type": item.get("asset_type"),
                    "note": item.get("note", ""),
                    "error": str(e),
                    "updated_at": updated_at,
                })

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
