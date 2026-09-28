import os
import json
import asyncio
import logging
import gc
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import yfinance as yf
import pandas as pd
import numpy as np

from scanner_charts import generate_chart_image
from signal_engine import get_signal_supertrend as get_signal
from datetime import timedelta
from json_utils import atomic_write_json

logger = logging.getLogger("scanner")


def _check_earnings_soon(symbol: str, days: int = 7) -> tuple[bool, Optional[str]]:
    """
    檢查該股票是否在 days 天內有財報。
    回傳 (earnings_soon: bool, earnings_date: str | None)。
    台股 yfinance 通常不回傳財報日，失敗時靜默回傳 (False, None)。
    """
    try:
        cal = yf.Ticker(symbol).calendar
        if cal is None or cal.empty:
            return False, None
        # calendar 是 DataFrame，欄位為日期，index 為項目名稱
        if "Earnings Date" in cal.index:
            ed = cal.loc["Earnings Date"]
            # 可能是 list 或單一值
            dates = ed if hasattr(ed, "__iter__") and not isinstance(ed, str) else [ed]
            today = datetime.now().date()
            for d in dates:
                try:
                    ed_date = pd.Timestamp(d).date()
                    delta = (ed_date - today).days
                    if -1 <= delta <= days:   # -1 允許昨天（財報剛出）
                        return True, str(ed_date)
                except Exception:
                    pass
    except Exception:
        pass
    return False, None

SCAN_RESULTS_DIR = Path(__file__).parent / "scan_results"
SCAN_CHARTS_DIR = Path(__file__).parent / "scan_charts"
SCAN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SCAN_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

US_SYMBOLS_BY_SECTOR = {
    "科技":     ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "IBM", "ORCL", "DELL", "HPQ"],
    "半導體":   ["AMD", "AVGO", "QCOM", "AMAT", "MU", "TXN", "TSM", "ARM", "MRVL", "SMCI", "LRCX", "KLAC", "NXPI", "ON"],
    "軟體":     ["CRM", "ADBE", "NOW", "INTU", "PLTR", "CRWD", "SNOW", "DDOG", "PANW", "ZS", "WDAY", "VEEV", "HUBS", "SHOP", "TEAM"],
    "通訊":     ["NFLX", "DIS", "CMCSA", "SPOT"],
    "平台經濟": ["UBER", "ABNB", "DASH"],
    "電商消費": ["AMZN", "TSLA", "NKE", "MCD", "SBUX", "COST", "HD", "TGT", "LULU"],
    "金融":     ["JPM", "BAC", "GS", "V", "MA", "MS", "PYPL", "BRK-B", "WFC", "C", "AXP", "BLK", "SCHW", "COF"],
    "醫療":     ["JNJ", "UNH", "LLY", "ABBV", "MRK", "GILD", "VRTX", "REGN", "AMGN", "BMY", "BIIB", "ISRG", "MDT", "BSX", "CVS", "CI"],
    "能源":     ["XOM", "CVX", "COP", "OXY", "SLB"],
    "工業":     ["CAT", "DE", "HON", "GE", "RTX", "LMT", "BA", "MMM"],
    "防禦公用": ["WMT", "PG", "KO", "PEP", "NEE", "SO"],
    "材料":     ["FCX", "NEM", "CF", "MOS"],
    "REITs":    ["AMT", "PLD", "O"],
    "加密概念": ["MSTR", "COIN", "HOOD"],
    "航太":     ["SPCX"],
}
US_SYMBOLS = [s for group in US_SYMBOLS_BY_SECTOR.values() for s in group]
US_SYMBOL_SECTOR = {s: sec for sec, syms in US_SYMBOLS_BY_SECTOR.items() for s in syms}


def get_twse_symbols() -> list:
    """從 TWSE OpenAPI 取得全部上市股票代號，回傳 ['2330.TW', ...] 格式。"""
    # 備用端點清單，依序嘗試
    endpoints = [
        "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",   # 上市公司基本資料
        "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",  # 每日成交
    ]
    for url in endpoints:
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list) or not data:
                continue
            symbols = []
            # 欄位名因端點而異，嘗試幾個常見的代號欄位
            code_keys = ["公司代號", "Code", "stock_no", "SecuritiesCompanyCode"]
            first = data[0]
            code_key = next((k for k in code_keys if k in first), None)
            if code_key is None:
                # 找值為 4 位數字的欄位
                code_key = next(
                    (k for k, v in first.items() if str(v).isdigit() and len(str(v)) == 4),
                    None,
                )
            if code_key is None:
                logger.warning(f"TWSE {url} 找不到代號欄位，跳過")
                continue
            for item in data:
                code = str(item.get(code_key, "")).strip()
                if code.isdigit() and len(code) == 4:
                    symbols.append(f"{code}.TW")
            if symbols:
                logger.info(f"TWSE 上市取得 {len(symbols)} 支（來源：{url}）")
                return symbols
        except Exception as e:
            logger.warning(f"TWSE 端點 {url} 失敗: {e}")
    logger.error("所有 TWSE 端點均失敗")
    return []


def get_tpex_symbols() -> list:
    """從 TPEX 取得全部上櫃股票代號，回傳 ['6547.TWO', ...] 格式。"""
    try:
        # 資料量約 4.5MB，需較長 timeout
        url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
        resp = requests.get(url, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        seen = set()
        symbols = []
        for item in data:
            code = str(item.get("SecuritiesCompanyCode", "")).strip()
            # 上櫃普通股：4 位純數字，排除 ETF/認購權證等
            if code.isdigit() and len(code) == 4 and code not in seen:
                seen.add(code)
                symbols.append(f"{code}.TWO")
        logger.info(f"TPEX 上櫃取得 {len(symbols)} 支")
        return symbols
    except Exception as e:
        logger.error(f"取得 TPEX 上櫃代號失敗: {e}")
        return []


def get_all_tw_symbols() -> list:
    """合併上市 (.TW) + 上櫃 (.TWO) 股票代號，失敗時回傳備援清單。"""
    twse = get_twse_symbols()
    tpex = get_tpex_symbols()
    if not twse:
        logger.warning("TWSE 上市清單為空，台股結果將只含上櫃")
    if not tpex:
        logger.warning("TPEX 上櫃清單為空")
    combined = twse + tpex
    if combined:
        logger.info(f"台股合計：上市 {len(twse)} + 上櫃 {len(tpex)} = {len(combined)} 支")
        return combined

    fallback = [
        "2330.TW","2317.TW","2454.TW","2308.TW","2382.TW",
        "2303.TW","2412.TW","2881.TW","2882.TW","1301.TW",
        "2886.TW","2891.TW","2002.TW","3008.TW","2357.TW",
        "6547.TWO","3045.TWO","5347.TWO","6409.TWO","3231.TWO",
    ]
    logger.warning(f"使用備援台股清單 ({len(fallback)} 支)")
    return fallback


def _build_tw_name_map() -> dict:
    """
    從 TWSE + TPEX API 建立 {symbol: 中文簡稱} 對照表。
    例：{"2330.TW": "台積電", "6547.TWO": "力旺"}
    失敗時回傳空 dict，呼叫端用代號本身作為 fallback。
    """
    name_map = {}
    # 上市（TWSE）
    try:
        resp = requests.get(
            "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
            timeout=15,
        )
        resp.encoding = "utf-8"
        for item in resp.json():
            code = str(item.get("公司代號", "")).strip()
            name = str(item.get("公司簡稱", "")).strip()
            if code.isdigit() and len(code) == 4 and name:
                name_map[f"{code}.TW"] = name
        logger.info(f"TWSE 名稱對照表：{len(name_map)} 筆")
    except Exception as e:
        logger.warning(f"TWSE 名稱抓取失敗: {e}")
    # 上櫃（TPEX）
    try:
        resp = requests.get(
            "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",
            timeout=60,
        )
        resp.encoding = "utf-8"
        seen = set()
        for item in resp.json():
            code = str(item.get("SecuritiesCompanyCode", "")).strip()
            name = str(item.get("CompanyName", "")).strip()
            if code.isdigit() and len(code) == 4 and name and code not in seen:
                seen.add(code)
                name_map[f"{code}.TWO"] = name
        logger.info(f"TPEX 名稱對照表累計：{len(name_map)} 筆")
    except Exception as e:
        logger.warning(f"TPEX 名稱抓取失敗: {e}")
    return name_map


MIN_AVG_VOL_TW = 500_000    # 台股：20日均量≥50萬股（約500張）
MIN_AVG_VOL_US = 500_000    # 美股：20日均量≥50萬股
MIN_PRICE_TW   = 10.0       # 台股：股價≥10元
MIN_PRICE_US   = 5.0        # 美股：股價≥5美元


def technical_prescreen(
    symbol: str,
    df: pd.DataFrame,
    min_avg_vol: float = MIN_AVG_VOL_TW,
    min_price:   float = MIN_PRICE_TW,
) -> Optional[dict]:
    """
    流動性過濾 + signal_engine 訊號評估。
    只有 signal = BUY 或 WATCH 的股票才進入圖表生成。
    """
    try:
        # ── 流動性門檻 ──────────────────────────────────────
        if len(df) < 20:
            return None
        close_last = float(df["Close"].iloc[-1])
        avg_vol_20 = float(df["Volume"].iloc[-20:].mean())

        # 過濾無效資料（NaN 收盤價或成交量）
        if np.isnan(close_last) or np.isnan(avg_vol_20):
            return None
        if close_last < min_price:
            return None
        if avg_vol_20 < min_avg_vol:
            return None

        # ── 訊號引擎 ────────────────────────────────────────
        result = get_signal(df)

        # 只保留 BUY / WATCH，WAIT 直接略過
        if result.get("signal") == "WAIT":
            return None

        # ── 財報日檢查（只對通過訊號的股票查詢，避免效能問題）──
        earnings_soon, earnings_date = _check_earnings_soon(symbol)

        ind = result.get("indicators", {})

        # 近期 15 根日 K（供 Claude 進場時機分析）
        recent = df.tail(15)
        ohlcv = []
        for ts, row in recent.iterrows():
            try:
                ohlcv.append({
                    "date":   str(ts)[:10],
                    "open":   round(float(row["Open"]),  2),
                    "high":   round(float(row["High"]),  2),
                    "low":    round(float(row["Low"]),   2),
                    "close":  round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]),
                })
            except Exception:
                pass

        return {
            "symbol":         symbol,
            "close":          round(close_last, 2),
            "change_pct":     ind.get("change_pct", 0.0),
            "rsi":            ind.get("rsi", 0.0),
            "volume_ratio":   ind.get("volume_ratio", 1.0),
            "avg_vol_20":     round(avg_vol_20),
            "trigger_reasons": result.get("trigger_reasons", []),
            "signal":         result.get("signal"),
            "signal_label":   result.get("signal_label"),
            "strategy":       result.get("strategy"),
            "regime":         result.get("regime"),
            "score":          result.get("score", 0),
            "trend_score":    result.get("trend_score", 0),
            "entry_score":    result.get("entry_score", 0),
            "pullback_valid": result.get("pullback_valid", False),
            "sl":             result.get("sl"),
            "tp":             result.get("tp"),
            "rr":             result.get("rr"),
            "add_on_levels":  result.get("add_on_levels", []),
            "reasons":        result.get("reasons", []),
            "indicators":     result.get("indicators", {}),
            "ohlcv":          ohlcv,
            "earnings_soon":  earnings_soon,
            "earnings_date":  earnings_date,
        }
    except Exception as e:
        logger.debug(f"快篩 {symbol} 失敗: {e}")
        return None


def _sanitize(obj):
    """遞迴將 dict/list 中的 NaN/Infinity 換成 None，避免 FastAPI JSON 序列化失敗。"""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj



def _fetch_batch_sync(symbols: list, period: str = "6mo") -> dict:
    result = {}
    try:
        raw = yf.download(
            symbols, period=period, group_by="ticker",
            auto_adjust=False, threads=True, progress=False,
        )
        is_multi = isinstance(raw.columns, pd.MultiIndex)
        for sym in symbols:
            try:
                df = raw[sym].copy() if is_multi else raw.copy()
                df.dropna(how="all", inplace=True)
                # 過濾週末列與零成交量列（yfinance 偶爾會回傳非交易日假資料）
                df = df[df.index.dayofweek < 5]
                df = df[df["Volume"] > 0]
                if len(df) >= 30:
                    result[sym] = df
            except Exception:
                pass
    except Exception as e:
        logger.error(f"批次下載失敗: {e}")
    return result


async def run_tw_scan() -> dict:
    """執行台股全掃描（上市 .TW + 上櫃 .TWO）：技術快篩 → 圖表生成。"""
    scan_time = datetime.now().isoformat()
    if datetime.now().weekday() >= 5:  # 週六=5, 週日=6
        logger.info("今日為週末，跳過台股掃描")
        return {"market": "tw", "scan_time": scan_time, "skipped": True,
                "message": "週末不進行掃描", "results": []}
    logger.info("開始台股全掃描...")

    symbols = get_all_tw_symbols()
    total_scanned = len(symbols)

    # 名稱對照表（與下載批次並行，不阻塞主流程）
    name_map = await asyncio.get_running_loop().run_in_executor(None, _build_tw_name_map)

    all_data = {}
    batch_size = 100
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i: i + batch_size]
        logger.info(f"下載 {i + 1}~{i + len(batch)} / {total_scanned}")
        batch_data = await asyncio.get_running_loop().run_in_executor(
            None, lambda b=batch: _fetch_batch_sync(b, "6mo")
        )
        all_data.update(batch_data)
        await asyncio.sleep(2)

    logger.info(f"成功下載 {len(all_data)} 支台股")

    candidates = []
    for sym, df in all_data.items():
        hit = technical_prescreen(sym, df)
        if hit:
            hit["sector"] = "台股"
            hit["name"] = name_map.get(sym, sym.split(".")[0])
            candidates.append((sym, df, hit))

    logger.info(f"技術快篩通過：{len(candidates)} 支")

    semaphore = asyncio.Semaphore(10)

    async def analyze_one(sym, df, prescreen_data):
        async with semaphore:
            chart_path = await asyncio.get_running_loop().run_in_executor(
                None, lambda s=sym, d=df: generate_chart_image(s, d)
            )
            return {**prescreen_data, "chart_path": chart_path}

    tasks = [analyze_one(sym, df, hit) for sym, df, hit in candidates]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results = [r for r in raw_results if isinstance(r, dict)]
    # 主排序：signal 優先（BUY > WATCH > WAIT），次排序：進場評分
    signal_order = {"BUY": 0, "WATCH": 1, "WAIT": 2}
    results.sort(key=lambda r: (
        signal_order.get(r.get("signal", "WAIT"), 2),
        -r.get("entry_score", 0),
    ))

    output = {
        "market": "tw",
        "scan_time": scan_time,
        "total_scanned": total_scanned,
        "pre_screened": len(candidates),
        "analyzed": len(results),
        "results": results,
    }
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = SCAN_RESULTS_DIR / f"tw_{date_str}.json"
    atomic_write_json(out_path, _sanitize(output), ensure_ascii=False, indent=2)

    del all_data
    gc.collect()

    try:
        from scan_tracker import update_tracker
        update_tracker(output)
    except Exception as e:
        logger.error(f"掃描追蹤器更新失敗: {e}")

    logger.info(f"台股掃描完成 → {out_path}")
    return output


async def run_us_scan() -> dict:
    """執行美股固定清單掃描。"""
    scan_time = datetime.now().isoformat()
    if datetime.now().weekday() >= 5:
        logger.info("今日為週末，跳過美股掃描")
        return {"market": "us", "scan_time": scan_time, "skipped": True,
                "message": "週末不進行掃描", "results": []}
    logger.info("開始美股掃描...")

    all_data = await asyncio.get_running_loop().run_in_executor(
        None, lambda: _fetch_batch_sync(US_SYMBOLS, "6mo")
    )
    logger.info(f"美股成功下載 {len(all_data)} 支")

    # 美股快篩：套用美股流動性門檻，只保留 BUY / WATCH
    candidates_us = []
    for sym, df in all_data.items():
        hit = technical_prescreen(sym, df, min_avg_vol=MIN_AVG_VOL_US, min_price=MIN_PRICE_US)
        if hit:
            hit["sector"] = US_SYMBOL_SECTOR.get(sym, "其他")
            hit["name"] = sym
            candidates_us.append((sym, df, hit))

    logger.info(f"美股技術快篩通過：{len(candidates_us)} / {len(all_data)} 支")

    semaphore = asyncio.Semaphore(10)

    async def analyze_one_us(sym, df, prescreen):
        async with semaphore:
            prescreen["sector"] = US_SYMBOL_SECTOR.get(sym, "其他")
            prescreen["name"] = sym
            chart_path = await asyncio.get_running_loop().run_in_executor(
                None, lambda s=sym, d=df: generate_chart_image(s, d)
            )
            return {**prescreen, "chart_path": chart_path}

    tasks = [analyze_one_us(sym, df, hit) for sym, df, hit in candidates_us]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results = [r for r in raw_results if isinstance(r, dict)]
    signal_order = {"BUY": 0, "WATCH": 1, "WAIT": 2}
    results.sort(key=lambda r: (
        signal_order.get(r.get("signal", "WAIT"), 2),
        -r.get("entry_score", 0),
    ))

    output = {
        "market": "us",
        "scan_time": scan_time,
        "total_scanned": len(US_SYMBOLS),
        "pre_screened": len(candidates_us),
        "analyzed": len(results),
        "results": results,
    }
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = SCAN_RESULTS_DIR / f"us_{date_str}.json"
    atomic_write_json(out_path, _sanitize(output), ensure_ascii=False, indent=2)

    try:
        from scan_tracker import update_tracker
        update_tracker(output)
    except Exception as e:
        logger.error(f"掃描追蹤器更新失敗: {e}")

    logger.info(f"美股掃描完成 → {out_path}")
    return output


def get_latest_scan_results(market: str) -> Optional[dict]:
    """讀取最新掃描結果。market: 'tw' 或 'us'"""
    try:
        files = sorted(SCAN_RESULTS_DIR.glob(f"{market}_*.json"), reverse=True)
        if not files:
            return None
        with open(files[0], "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"讀取掃描結果失敗: {e}")
        return None
