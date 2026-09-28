import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse

from scanner import get_latest_scan_results, run_tw_scan, run_us_scan, SCAN_CHARTS_DIR
from signal_engine import _supertrend

router = APIRouter(prefix="/api/scanner", tags=["scanner"])
logger = logging.getLogger("router_scanner")

_scanning_status = {"tw": False, "us": False}
_scan_error = {"tw": None, "us": None}


async def _run_scan_background(market: str):
    """背景執行掃描，不阻塞 HTTP 連線"""
    try:
        _scanning_status[market] = True
        _scan_error[market] = None
        if market == "tw":
            await run_tw_scan()
        else:
            await run_us_scan()
        logger.info(f"{market.upper()} 背景掃描完成")
    except Exception as e:
        _scan_error[market] = str(e)
        logger.error(f"{market.upper()} 背景掃描失敗: {e}", exc_info=True)
    finally:
        _scanning_status[market] = False


@router.get("/results")
async def get_scan_results(market: str = Query("tw", description="tw 或 us")):
    if market not in ("tw", "us"):
        raise HTTPException(status_code=400, detail="market 必須為 'tw' 或 'us'")
    data = get_latest_scan_results(market)
    if data is None:
        return {"results": [], "message": "尚無掃描結果，請先執行掃描", "market": market,
                "total_scanned": 0, "pre_screened": 0, "analyzed": 0}

    # 合併 Claude 分析結果
    try:
        from claude_scanner_analysis import get_latest_claude_results
        claude_data = get_latest_claude_results(market)
        if claude_data:
            claude_map = {
                r["symbol"]: {
                    "claude_analysis": r.get("claude_analysis"),
                    "signal_history":  r.get("signal_history", []),
                }
                for r in claude_data.get("results", [])
            }
            for result in data.get("results", []):
                entry = claude_map.get(result["symbol"]) or {}
                result["claude_analysis"] = entry.get("claude_analysis")
                result["signal_history"]  = entry.get("signal_history", [])
    except Exception as e:
        logger.warning(f"合併 Claude 分析失敗: {e}")

    return data


@router.post("/run")
async def trigger_scan(
    background_tasks: BackgroundTasks,
    market: str = Query("tw", description="tw 或 us"),
):
    """
    立即觸發掃描（背景執行，立即回傳）。
    前端應輪詢 GET /api/scanner/status 確認掃描進度。
    """
    if market not in ("tw", "us"):
        raise HTTPException(status_code=400, detail="market 必須為 'tw' 或 'us'")
    if _scanning_status.get(market):
        return {"message": f"{market.upper()} 掃描已在進行中，請稍候", "scanning": True, "started": False}

    background_tasks.add_task(_run_scan_background, market)
    return {"message": f"{market.upper()} 掃描已啟動，請輪詢 /api/scanner/status 確認進度", "scanning": True, "started": True}


@router.get("/status")
async def get_scan_status():
    tw_data = get_latest_scan_results("tw")
    us_data = get_latest_scan_results("us")
    return {
        "tw": {
            "last_scan_time": tw_data.get("scan_time") if tw_data else None,
            "signal_count": len(tw_data.get("results", [])) if tw_data else 0,
            "is_scanning": _scanning_status.get("tw", False),
        },
        "us": {
            "last_scan_time": us_data.get("scan_time") if us_data else None,
            "signal_count": len(us_data.get("results", [])) if us_data else 0,
            "is_scanning": _scanning_status.get("us", False),
        },
        "schedule": {
            "tw": "週一至週五 08:00（台灣時間）",
            "us": "週一至週五 21:00（台灣時間）",
        },
    }


@router.get("/market-state")
async def get_market_state():
    """回傳加權指數（^TWII）與 S&P500（^GSPC）的三重 ST 狀態。"""
    ST_PARAMS = [(11, 2.0), (10, 1.0), (12, 3.0)]

    def calc_state(ticker_sym: str):
        try:
            df = yf.Ticker(ticker_sym).history(period="6mo", auto_adjust=True)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[["High", "Low", "Close"]].dropna()
            if len(df) < 60:
                return None
            h = df["High"].values.astype(float)
            l = df["Low"].values.astype(float)
            c = df["Close"].values.astype(float)
            dirs = []
            for p, m in ST_PARAMS:
                d, _ = _supertrend(h, l, c, p, m)
                dirs.append(int(d[-1]))
            green = sum(1 for d in dirs if d == 1)
            return {
                "green_count": green,
                "directions": dirs,
                "close": round(float(c[-1]), 2),
                "date": df.index[-1].strftime("%Y-%m-%d"),
            }
        except Exception as e:
            logger.warning(f"market-state {ticker_sym} 失敗: {e}")
            return None

    import datetime
    month = datetime.date.today().month
    in_season = month not in {5, 6, 7, 8, 9}  # 10–4月為旺季

    tw = calc_state("^TWII")
    us = calc_state("^GSPC")
    return {"tw": tw, "us": us, "month": month, "in_season": in_season}


@router.get("/chart/{symbol}")
async def get_chart(symbol: str):
    safe_symbol = symbol.replace(".", "_").replace("/", "_")
    matches = sorted(SCAN_CHARTS_DIR.glob(f"{safe_symbol}_*.png"), reverse=True)
    if not matches:
        raise HTTPException(status_code=404, detail=f"找不到 {symbol} 的圖表")
    return FileResponse(str(matches[0]), media_type="image/png")


# ── 個股歷史訊號查詢 ────────────────────────────────────────────────────────────

@router.get("/history/symbol/{symbol}")
async def get_symbol_history(symbol: str, market: str = Query("tw")):
    """查詢某支股票在所有歷史掃描中的出現紀錄"""
    from pathlib import Path
    import json as _json

    symbol = symbol.upper()
    results_dir = Path(__file__).parent.parent / "scan_results"
    # 只讀主掃描檔（不含 claude_ 前綴）
    files = sorted(results_dir.glob(f"{market}_*.json"), reverse=True)

    records = []
    for f in files:
        try:
            data = _json.loads(f.read_text(encoding="utf-8"))
            if data.get("skipped"):
                continue
            for r in data.get("results", []):
                if r.get("symbol", "").upper() == symbol:
                    records.append({
                        "date":           data.get("scan_time", f.stem.replace(f"{market}_", ""))[:10],
                        "signal":         r.get("signal"),
                        "close":          r.get("close"),
                        "change_pct":     r.get("change_pct"),
                        "rsi":            r.get("rsi"),
                        "volume_ratio":   r.get("volume_ratio"),
                        "trend_score":    r.get("trend_score"),
                        "entry_score":    r.get("entry_score"),
                        "trigger_reasons": r.get("trigger_reasons", []),
                    })
                    break
        except Exception:
            continue

    return {"symbol": symbol, "market": market, "records": records}


# ── 每日總結 ──────────────────────────────────────────────────────────────────

@router.get("/daily-summary")
async def get_daily_summary():
    """取得最新每日盤後總結（台美股精選 + 市場觀察）"""
    from claude_scanner_analysis import get_latest_daily_summary
    result = get_latest_daily_summary()
    if not result:
        return {"summary": None, "date": None, "generated": None}
    return result


# ── 掃描追蹤器 ────────────────────────────────────────────────────────────────

@router.get("/tracker/open")
async def get_tracker_open(market: str = Query(None, description="tw / us / 不填=全部")):
    """取得目前開放追蹤部位"""
    from scan_tracker import get_open_positions
    return {"positions": get_open_positions(market)}


@router.get("/tracker/closed")
async def get_tracker_closed(
    market: str = Query(None, description="tw / us / 不填=全部"),
    limit:  int = Query(100, description="最多返回幾筆"),
):
    """取得已關閉部位（含損益記錄）"""
    from scan_tracker import get_closed_positions
    return {"positions": get_closed_positions(market, limit)}


@router.get("/tracker/summary")
async def get_tracker_summary():
    """取得掃描追蹤整體績效摘要"""
    from scan_tracker import get_summary
    return get_summary()


@router.delete("/tracker/open/{symbol}")
async def close_tracker_position(symbol: str, reason: str = Query("手動出場")):
    """手動關閉追蹤部位（用最新收盤價計算損益）"""
    import yfinance as yf
    from scan_tracker import _load, _save, _close_position
    from datetime import datetime

    symbol = symbol.upper()
    data   = _load()
    open_p = data["open"]
    target = next((p for p in open_p if p["symbol"] == symbol), None)
    if not target:
        raise HTTPException(status_code=404, detail=f"找不到開放部位 {symbol}")

    # 嘗試取得最新收盤價
    try:
        df = yf.Ticker(symbol).history(period="2d", auto_adjust=False)
        exit_price = float(df["Close"].iloc[-1])
    except Exception:
        exit_price = target.get("current_price", target["entry_price"])

    today  = datetime.now().strftime("%Y-%m-%d")
    closed = _close_position(target, exit_price, today, reason)
    data["closed"].append(closed)
    data["open"] = [p for p in open_p if p["symbol"] != symbol]
    _save(data)

    return {"message": f"已關閉 {symbol}，PnL {closed['pnl_pct']:.2f}%", "record": closed}
