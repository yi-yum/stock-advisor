"""
每日掃描 Claude 分析工具模組（不需要 Anthropic API Key）

由 Claude Code 排程任務呼叫：
  get_scan_candidates(market, date)     → 取得待分析股票清單
  build_buy_prompt(stock, market)       → BUY 完整分析提示詞（含 S/R）
  build_watch_prompt(stock, market)     → WATCH 精簡分析提示詞
  save_claude_analysis(market, results, meta, date)  → 儲存分析結果
  get_latest_claude_results(market)     → 讀取最新結果（前端/API 用）
"""
import json
import re
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

from json_utils import atomic_write_json

logger = logging.getLogger("claude_scanner_analysis")
SCAN_RESULTS_DIR   = Path(__file__).parent / "scan_results"
BACKTEST_STATS_DIR = Path(__file__).parent / "backtest_results"


# ── 歷史回測績效 ──────────────────────────────────────────────────────────────

def build_backtest_stats(market: str = "tw") -> dict:
    """
    從回測 CSV 計算每支股票的歷史三重ST訊號統計。
    回傳 {symbol: {n, win_rate, avg_win, avg_loss, avg_days, ev}} 。
    """
    pattern = f"signals_{market}_*.csv"
    files = sorted(BACKTEST_STATS_DIR.glob(pattern), reverse=True)
    # 優先用未加過濾條件的完整版（檔名最短）
    files = [f for f in files if "_taiex" not in f.stem and "_season" not in f.stem]
    if not files:
        return {}
    try:
        df = pd.read_csv(files[0])
    except Exception as e:
        logger.warning(f"無法讀取回測CSV: {e}")
        return {}

    stats = {}
    for sym, g in df.groupby("symbol"):
        n      = len(g)
        wins   = g[g["return_pct"] > 0]
        losses = g[g["return_pct"] <= 0]
        stats[sym] = {
            "n":        n,
            "win_rate": round((len(wins) / n) * 100),
            "avg_win":  round(wins["return_pct"].mean(), 1)  if len(wins)   > 0 else 0,
            "avg_loss": round(losses["return_pct"].mean(), 1) if len(losses) > 0 else 0,
            "avg_days": round(g["days_held"].mean()),
            "ev":       round(g["return_pct"].mean(), 1),
        }
    return stats


_BACKTEST_STATS_CACHE: dict = {}
_MARGIN_MAP_CACHE:    dict = {}
_EARNINGS_CACHE:      dict = {}


# ── 美股財報日期 ──────────────────────────────────────────────────────────────────

def _get_earnings_info(symbol: str) -> dict | None:
    """
    取得美股下次財報日期（yfinance calendar）。
    回傳 {"date": "YYYY-MM-DD", "days": int} 或 None（無資料 / 已過期）。
    結果快取於 _EARNINGS_CACHE，同一 session 不重複查詢。
    """
    global _EARNINGS_CACHE
    if symbol in _EARNINGS_CACHE:
        return _EARNINGS_CACHE[symbol]

    result = None
    try:
        import yfinance as yf
        from datetime import date as _date
        cal = yf.Ticker(symbol).calendar
        if cal:
            earnings_dates = cal.get("Earnings Date", [])
            now = _date.today()
            future = []
            for d in earnings_dates:
                try:
                    d_date = d.date() if hasattr(d, "date") else d
                    if d_date >= now:
                        future.append(d_date)
                except Exception:
                    pass
            if future:
                next_date = min(future)
                days_until = (next_date - now).days
                result = {"date": next_date.strftime("%Y-%m-%d"), "days": days_until}
    except Exception:
        pass

    _EARNINGS_CACHE[symbol] = result
    return result


def _load_backtest_stats(market: str) -> dict:
    """載入回測統計（有快取則直接回傳，避免重複讀檔）。"""
    global _BACKTEST_STATS_CACHE
    if market not in _BACKTEST_STATS_CACHE:
        _BACKTEST_STATS_CACHE[market] = build_backtest_stats(market)
    return _BACKTEST_STATS_CACHE[market]


def _fmt_backtest_stats(stats: dict | None) -> str:
    """將單支股票的統計格式化為 prompt 字串。"""
    if not stats or stats.get("n", 0) < 3:
        n = stats.get("n", 0) if stats else 0
        return f"歷史資料不足（回測期間僅 {n} 次訊號）"
    n        = stats["n"]
    wr       = stats["win_rate"]
    avg_win  = stats["avg_win"]
    avg_loss = stats["avg_loss"]
    avg_days = stats["avg_days"]
    ev       = stats["ev"]
    quality  = "優" if ev > 5 else "良" if ev > 2 else "普通" if ev > 0 else "偏差"
    return (
        f"回測期間（2021–2026）共觸發 {n} 次三重ST翻多訊號\n"
        f"勝率：{wr}%　期望值：{ev:+.1f}%（信號品質：{quality}）\n"
        f"平均獲利（贏）：{avg_win:+.1f}%　平均虧損（輸）：{avg_loss:+.1f}%\n"
        f"平均持倉天數：{avg_days} 天"
    )


# ── 融資融券資料 ──────────────────────────────────────────────────────────────────

def _build_margin_map(date_str: str) -> dict:
    """
    一次呼叫 TWSE 融資融券 API（MI_MARGN），回傳全市場個股融資融券對照表。
    只發一次 HTTP 請求，後續查詢使用快取。

    回傳格式：
    {"2330": {"margin_balance": 50000, "margin_change": +500, "short_balance": 3000, "short_change": -100}}

    實際 API 結構（tables[1].data，16 欄）：
    [0]=代號 [1]=名稱 [2]=融資買進 [3]=融資賣出 [4]=現金償還
    [5]=前日融資餘額 [6]=今日融資餘額 [7]=限額
    [8]=融券賣出 [9]=融券買進 [10]=現券償還
    [11]=前日融券餘額 [12]=今日融券餘額 [13]=限額 [14]=資券相抵 [15]=備註
    """
    global _MARGIN_MAP_CACHE
    if date_str in _MARGIN_MAP_CACHE:
        return _MARGIN_MAP_CACHE[date_str]

    try:
        import requests
        from datetime import datetime as _dt, timedelta

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        url = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"

        # 嘗試當天及前4個交易日（跳過週末）
        base = _dt.strptime(date_str, "%Y%m%d")
        candidates = []
        d = base
        while len(candidates) < 5:
            if d.weekday() < 5:
                candidates.append(d.strftime("%Y%m%d"))
            d -= timedelta(days=1)

        def _to_int(s):
            try:
                return int(str(s).replace(",", "").strip())
            except (ValueError, AttributeError):
                return 0

        for try_date in candidates:
            try:
                params = {"date": try_date, "selectType": "ALL", "response": "json"}
                resp = requests.get(url, params=params, headers=headers, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                # 個股明細在 tables[1]
                tables = data.get("tables", [])
                if len(tables) < 2:
                    continue
                rows = tables[1].get("data", [])
                if not rows:
                    continue
                result = {}
                for row in rows:
                    if len(row) < 13:
                        continue
                    code = str(row[0]).strip()
                    # 只取純數字代號（排除 ETF 如 00400A）
                    if not code[:4].isdigit():
                        continue
                    mb_prev  = _to_int(row[5])
                    mb_today = _to_int(row[6])
                    sb_prev  = _to_int(row[11])
                    sb_today = _to_int(row[12])
                    result[code] = {
                        "margin_balance": mb_today,
                        "margin_change":  mb_today - mb_prev,
                        "short_balance":  sb_today,
                        "short_change":   sb_today - sb_prev,
                    }
                if result:
                    _MARGIN_MAP_CACHE[date_str] = result
                    return result
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"融資融券地圖建立失敗: {e}")

    _MARGIN_MAP_CACHE[date_str] = {}
    return {}


# ── 提示詞建構 ────────────────────────────────────────────────────────────────

def _fmt_ohlcv(ohlcv: list, avg_vol: float = 0) -> str:
    """將 OHLCV 列表格式化為緊湊文字表（最多 15 根，含相對成交量）"""
    if not ohlcv:
        return "（無K線資料）"
    lines = ["日期          開盤     高點     低點     收盤    漲跌    量比"]
    prev_close = None
    for bar in ohlcv[-15:]:
        chg = "     "
        if prev_close and prev_close > 0:
            pct = (bar['close'] - prev_close) / prev_close * 100
            chg = f"{pct:+.1f}%"
        vol_str = ""
        if avg_vol and avg_vol > 0:
            vr = bar['volume'] / avg_vol
            vol_str = f"  {vr:.1f}x"
        lines.append(
            f"{bar['date']}  {bar['open']:>7}  {bar['high']:>7}  "
            f"{bar['low']:>7}  {bar['close']:>7}  {chg:>6}{vol_str}"
        )
        prev_close = bar['close']
    return "\n".join(lines)


def build_buy_prompt(stock: dict, market: str) -> str:
    """BUY 訊號分析提示詞（聚焦假突破判斷，隔日開盤進場決策）"""
    market_label = '台股' if market == 'tw' else '美股'
    ind   = stock.get('indicators', {})
    close = stock.get('close')
    sl    = stock.get('sl')
    tp    = stock.get('tp')
    atr   = ind.get('atr', 0)

    ema20   = ind.get('ema20')
    ema50   = ind.get('ema50')
    ema200  = ind.get('ema200')
    high20  = ind.get('high20')
    low20   = ind.get('low20')
    high60  = ind.get('high60')
    low60   = ind.get('low60')

    def _fmt(v): return round(v, 2) if v else '—'

    sr_data = (
        f"- EMA20：{_fmt(ema20)}　EMA50：{_fmt(ema50)}　EMA200：{_fmt(ema200)}\n"
        f"- 20日高點：{_fmt(high20)}　20日低點：{_fmt(low20)}\n"
        f"- 60日高點：{_fmt(high60)}　60日低點：{_fmt(low60)}\n"
        f"- ATR(14)：{_fmt(atr)}\n"
        f"- ST動態支撐（失效線）：{sl if sl else '—'}"
    )

    reasons_text = '\n'.join(f'  {r}' for r in stock.get('reasons', []))
    sl_hint = f"{sl}" if sl else "依ST支撐線判斷"
    tp_hint = f"{tp}" if tp else "依壓力位判斷"
    avg_vol = stock.get('avg_vol_20', 0)
    ohlcv_table = _fmt_ohlcv(stock.get('ohlcv', []), avg_vol)
    name = stock.get('name', '')
    name_display = f"（{name}）" if name and name != stock.get('symbol', '') else ''

    chip_rank = stock.get('chip_rank')
    chip_net  = stock.get('chip_net')
    chip_foreign = stock.get('chip_foreign')
    chip_trust   = stock.get('chip_trust')
    if chip_rank and chip_net is not None:
        net_wan = round(chip_net / 1000, 1)
        f_wan   = round(chip_foreign / 1000, 1) if chip_foreign else 0
        t_wan   = round(chip_trust   / 1000, 1) if chip_trust   else 0
        chip_section = (
            f"\n【籌碼面】\n"
            f"三大法人買超排名：第 {chip_rank} 名　今日合計買超：{net_wan} 萬張\n"
            f"外資：{f_wan:+.1f} 萬張　投信：{t_wan:+.1f} 萬張"
        )
        if chip_rank <= 30:
            chip_section += "\n⭐ 籌碼共振：技術面翻多 + 法人大買超，雙重訊號確認"
    else:
        chip_section = ""

    margin = stock.get('margin_data')
    if margin:
        mc = margin['margin_change']
        sc = margin['short_change']
        margin_section = (
            f"\n【融資融券】\n"
            f"融資餘額：{margin['margin_balance']:,} 張（今日增減：{mc:+,} 張）"
            f"　融券餘額：{margin['short_balance']:,} 張（今日增減：{sc:+,} 張）"
        )
    else:
        margin_section = ""

    signal_history = stock.get('signal_history', [])
    _today = datetime.now().strftime("%Y%m%d")
    history_text = _fmt_signal_history(signal_history, "BUY", _today)
    history_section = f"\n【歷史訊號脈絡】\n{history_text}"

    bt_stats_text = _fmt_backtest_stats(stock.get('backtest_stats'))
    bt_section = f"\n【歷史回測績效（本股三重ST訊號）】\n{bt_stats_text}"

    chip_row = "\n- 籌碼：法人方向是否配合（台股）" if market == "tw" else ""

    earnings_section = ""
    if market == "us":
        ei = _get_earnings_info(stock.get("symbol", ""))
        if ei:
            days = ei["days"]
            date_str_e = ei["date"]
            if days <= 7:
                warn = "⚠️ 財報極近，強烈建議跳過本次訊號"
            elif days <= 14:
                warn = "⚠️ 財報在即，嚴格控制倉位或等財報後再進"
            elif days <= 30:
                warn = "注意財報風險，務必設置止損"
            else:
                warn = "財報風險低"
            earnings_section = f"\n\n【財報風險】距下次財報：{days} 天（{date_str_e}）　{warn}"

    return f"""你是資深{market_label}技術分析師，專精「三重 SuperTrend」趨勢追蹤策略。
訊號用途：隔日開盤進場，核心問題是「今日突破是真是假？」。

【重要：輸出格式規定】所有價格只寫數字，勿加貨幣符號。

【今日訊號資料】
代號：{stock.get('symbol')}{name_display}　板塊：{stock.get('sector', '')}
收盤：{close}　漲跌：{stock.get('change_pct', 0):+.1f}%
RSI：{stock.get('rsi')}　量比：{stock.get('volume_ratio', 1):.1f}x{chip_section}{margin_section}{history_section}{bt_section}{earnings_section}

【SuperTrend 狀態】三條全部翻多（今日觸發）
{reasons_text}

【近期K線走勢（最近15根日K）】
{ohlcv_table}

【量化參考數據】
{sr_data}

請依照以下格式輸出分析（繁體中文，語氣專業直接）：

【突破背景】
說明今日是「整理後首次突破」還是「連漲後翻多」，近期K線型態（盤整突破 / V型反彈 / 追高），以及本股歷史EV/勝率是否支持進場。

【假突破風險評估】低 / 中 / 高
逐項判斷：
- 量能：今日量比（>1.5x 理想，<1.0x 為警訊）
- K線型態：今日實體大小、是否有明顯上影線
- RSI：是否偏高（>75 為警訊）
- 近期背景：連漲幾日後觸發（3日以上風險偏高）{chip_row}
綜合結論：低 / 中 / 高

【明日開盤建議】🟢 / 🟡 / 🔴
選一種並說明具體條件：
- 🟢 直接進場：假突破風險低，開盤可介入
- 🟡 等確認再進：說明要看什麼（如：開盤後守住 XXX 以上再買）
- 🔴 跳過本次：說明原因（EV偏差 / 假突破風險高 / 連漲過多）

【失效條件】
收盤跌破 {sl_hint} 視為假突破，出場不等
（止盈參考：{tp_hint}）"""


def build_watch_prompt(stock: dict, market: str) -> str:
    """WATCH 訊號分析提示詞（三重 SuperTrend 2/3，重點評估翻多可能性與翻多後品質）"""
    ind      = stock.get('indicators', {})
    ema20    = ind.get('ema20')
    ema50    = ind.get('ema50')
    low20    = ind.get('low20')
    high20   = ind.get('high20')
    atr      = ind.get('atr', 0)
    rsi      = stock.get('rsi', ind.get('rsi'))
    vr       = stock.get('volume_ratio', ind.get('volume_ratio', 1))
    close    = stock.get('close')
    st_dirs  = ind.get('st_directions', [])
    reasons_text = '\n'.join(f'  {r}' for r in stock.get('reasons', []))

    def _fmt(v): return round(v, 2) if v else '—'

    _ST_PARAMS = [(11, 2.0), (10, 1.0), (12, 3.0)]
    red_lines = []
    for i, (p, m) in enumerate(_ST_PARAMS):
        if i < len(st_dirs) and st_dirs[i] != 1:
            red_lines.append(f"ST({p},{m})")
    red_desc = "、".join(red_lines) if red_lines else "（無）"
    avg_vol = stock.get('avg_vol_20', 0)
    ohlcv_table = _fmt_ohlcv(stock.get('ohlcv', []), avg_vol)
    name = stock.get('name', '')
    name_display = f"（{name}）" if name and name != stock.get('symbol', '') else ''

    chip_rank = stock.get('chip_rank')
    chip_net  = stock.get('chip_net')
    chip_foreign = stock.get('chip_foreign')
    chip_trust   = stock.get('chip_trust')
    if chip_rank and chip_net is not None:
        net_wan = round(chip_net / 1000, 1)
        f_wan   = round(chip_foreign / 1000, 1) if chip_foreign else 0
        t_wan   = round(chip_trust   / 1000, 1) if chip_trust   else 0
        chip_section = (
            f"\n【籌碼面】\n"
            f"三大法人買超排名：第 {chip_rank} 名　今日合計買超：{net_wan} 萬張\n"
            f"外資：{f_wan:+.1f} 萬張　投信：{t_wan:+.1f} 萬張"
        )
        if chip_rank <= 30:
            chip_section += "\n⭐ 籌碼共振：技術面 2/3 翻多 + 法人大買超，翻多可能性加分"
    else:
        chip_section = ""

    margin = stock.get('margin_data')
    if margin:
        mc = margin['margin_change']
        sc = margin['short_change']
        margin_section = (
            f"\n【融資融券】\n"
            f"融資餘額：{margin['margin_balance']:,} 張（今日增減：{mc:+,} 張）"
            f"　融券餘額：{margin['short_balance']:,} 張（今日增減：{sc:+,} 張）"
        )
    else:
        margin_section = ""

    signal_history = stock.get('signal_history', [])
    _today = datetime.now().strftime("%Y%m%d")
    history_text = _fmt_signal_history(signal_history, "WATCH", _today)
    history_section = f"\n【歷史訊號脈絡】\n{history_text}"

    bt_stats_text = _fmt_backtest_stats(stock.get('backtest_stats'))
    bt_section = f"\n【歷史回測績效（本股三重ST訊號）】\n{bt_stats_text}"

    chip_row = "\n- 籌碼方向（台股）" if market == "tw" else ""

    earnings_section_w = ""
    if market == "us":
        ei_w = _get_earnings_info(stock.get("symbol", ""))
        if ei_w:
            days_w = ei_w["days"]
            date_str_w = ei_w["date"]
            if days_w <= 7:
                warn_w = "⚠️ 財報極近，即使翻多也建議跳過"
            elif days_w <= 14:
                warn_w = "⚠️ 財報在即，翻多後倉位需嚴格控制"
            elif days_w <= 30:
                warn_w = "注意財報風險"
            else:
                warn_w = "財報風險低"
            earnings_section_w = f"\n\n【財報風險】距下次財報：{days_w} 天（{date_str_w}）　{warn_w}"

    return f"""你是資深技術分析師，專精三重 SuperTrend 策略。
訊號用途：等待第三條ST翻多進入BUY後隔日進場，先評估翻多機率與翻多後的訊號品質。

【重要：輸出格式規定】所有價格只寫數字，不加貨幣符號。

【WATCH 訊號資料】
代號：{stock.get('symbol')}{name_display}　板塊：{stock.get('sector', '')}
收盤：{close}　漲跌：{stock.get('change_pct', 0):+.1f}%
RSI：{_fmt(rsi)}　量比：{_fmt(vr)}x
EMA20：{_fmt(ema20)}　EMA50：{_fmt(ema50)}
20日區間：{_fmt(low20)} ~ {_fmt(high20)}
ATR(14)：{_fmt(atr)}{chip_section}{margin_section}{history_section}{bt_section}{earnings_section_w}

【三重 SuperTrend 狀態】2/3 已翻多，待第三條確認
{reasons_text}
⚠️ 尚未翻多的 ST：{red_desc}

【近期K線走勢（最近15根日K）】
{ohlcv_table}

請依照以下格式輸出分析（繁體中文，語氣專業直接）：

【目前缺口】
說明 {red_desc} 仍為空方的原因：距ST壓力線大約幾個ATR、目前趨勢動能強弱。

【翻多可能性評估】高 / 中 / 低
逐項判斷：
- RSI動能方向（上升 / 橫盤 / 下降）
- 量能配合度（近期是否有放量傾向）
- 近期K線型態（連漲 / 整理 / 回測支撐）
- 距ST壓力估算（幾個ATR）{chip_row}
綜合結論：高 / 中 / 低

【翻多所需條件】
具體說明：需突破哪個價位、量能需達多少

【翻多後的假突破風險】低 / 中 / 高
若第三條ST翻多觸發BUY，評估該訊號的可靠度（依據EV/WR、當下RSI位置、近期漲幅背景）

【操作建議】觀望等待 / 等翻多確認後進場 / 不建議介入
【風險評級】低 / 中 / 高"""


# ── 籌碼資料 ──────────────────────────────────────────────────────────────────

SECTOR_FLOW_DIR = Path(__file__).parent / "sector_flow_data"

def _build_chip_map(date_str: str) -> dict:
    """
    讀取台股板塊流向 JSON，回傳個股三大法人排名對照表。
    格式：{"2330": {"rank": 1, "total_net": 89279, "foreign_net": 77685, "trust_net": 3803}}
    找不到檔案或解析失敗時回傳空 dict。
    """
    flow_file = SECTOR_FLOW_DIR / f"tw_{date_str}.json"
    if not flow_file.exists():
        # 嘗試最近一份
        files = sorted(SECTOR_FLOW_DIR.glob("tw_*.json"), reverse=True)
        if not files:
            return {}
        flow_file = files[0]

    try:
        with open(flow_file, encoding="utf-8") as f:
            data = json.load(f)
        # 合併所有板塊的個股
        all_stocks = []
        for sector in data.get("sectors", []):
            for s in sector.get("stocks", []):
                code = str(s.get("code", "")).strip()
                if code:
                    all_stocks.append({
                        "code":        code,
                        "total_net":   s.get("total_net", 0),
                        "foreign_net": s.get("foreign_net", 0),
                        "trust_net":   s.get("trust_net", 0),
                    })
        # 去重（同代號只保留 total_net 最大的）
        seen = {}
        for s in all_stocks:
            code = s["code"]
            if code not in seen or s["total_net"] > seen[code]["total_net"]:
                seen[code] = s
        # 依三大合計排序（買超多的排前面）
        ranked = sorted(seen.values(), key=lambda x: x["total_net"], reverse=True)
        return {
            s["code"]: {
                "rank":        i + 1,
                "total_net":   s["total_net"],
                "foreign_net": s["foreign_net"],
                "trust_net":   s["trust_net"],
            }
            for i, s in enumerate(ranked)
        }
    except Exception as e:
        logger.warning(f"籌碼地圖建立失敗: {e}")
        return {}


# ── 歷史訊號 ──────────────────────────────────────────────────────────────────

def _build_signal_history_map(market: str, current_date_str: str, days: int = 20) -> dict:
    """
    讀取最近 days 天的掃描 JSON（不含當天），回傳各 symbol 的訊號序列。
    格式：{"2330.TW": [{"date": "20260918", "signal": "WATCH"}, ...]}  最舊→最新
    """
    files = sorted(
        [f for f in SCAN_RESULTS_DIR.glob(f"{market}_*.json")
         if not f.name.startswith("claude_") and f.stem.split("_")[1] < current_date_str],
        reverse=True,
    )[:days]
    files = list(reversed(files))  # 改成舊→新

    history: dict = {}
    for f in files:
        date = f.stem.split("_")[1]
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            for r in data.get("results", []):
                sym = r.get("symbol")
                sig = r.get("signal")
                if sym and sig in ("BUY", "WATCH"):
                    history.setdefault(sym, []).append({"date": date, "signal": sig})
        except Exception:
            pass
    return history


def _fmt_signal_history(history: list, current_signal: str, current_date: str) -> str:
    """
    將歷史訊號序列格式化為 prompt 字串。
    history: [{"date": "20260918", "signal": "WATCH"}, ...]  舊→新（不含當天）
    """
    if not history:
        return f"首次出現（過去20個交易日無此股票訊號）"

    # 格式化每一天
    parts = []
    for h in history:
        d = h["date"]
        label = f"{d[4:6]}/{d[6:8]}"
        parts.append(f"{label} {h['signal']}")
    today = f"{current_date[4:6]}/{current_date[6:8]}"
    parts.append(f"{today} {current_signal}（今日）")
    chain = " → ".join(parts)

    # 產生摘要
    all_signals = [h["signal"] for h in history]
    prev = history[-1]["signal"] if history else None
    watch_streak = 0
    for h in reversed(history):
        if h["signal"] == "WATCH":
            watch_streak += 1
        else:
            break

    summary_parts = []
    if current_signal == "BUY" and prev == "WATCH":
        summary_parts.append(f"WATCH 升級 BUY（連續觀察 {watch_streak} 天後確認翻多）" if watch_streak > 1
                             else "昨日 WATCH → 今日升級 BUY")
    elif current_signal == "BUY" and prev == "BUY":
        summary_parts.append("BUY 訊號持續")
    elif current_signal == "WATCH" and watch_streak > 0:
        summary_parts.append(f"WATCH 持續 {watch_streak + 1} 天（含今日），尚未突破")
    elif current_signal == "WATCH" and prev is None:
        summary_parts.append("今日首次出現 WATCH")

    summary = "　".join(summary_parts) if summary_parts else ""
    return f"{chain}\n{summary}" if summary else chain


# ── 資料讀取 ──────────────────────────────────────────────────────────────────

def get_scan_candidates(market: str, date_str: str = None) -> dict:
    """
    讀取最新掃描 JSON，回傳待分析的 BUY/WATCH 候選股清單。

    Returns:
        {
            "buy":      [...],   # BUY 股票清單
            "watch":    [...],   # WATCH 股票清單
            "meta":     {...},   # scan_time, total_scanned, pre_screened
            "date_str": "YYYYMMDD",
            "source_file": "...",
            "error": None
        }
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    scan_file = SCAN_RESULTS_DIR / f"{market}_{date_str}.json"
    if not scan_file.exists():
        files = sorted(
            [f for f in SCAN_RESULTS_DIR.glob(f"{market}_*.json")
             if not f.name.startswith("claude_")],
            reverse=True,
        )
        if not files:
            return {"buy": [], "watch": [], "error": f"找不到 {market} 掃描結果"}
        scan_file = files[0]
        date_str = scan_file.stem.split("_")[1]

    with open(scan_file, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    meta = {
        "scan_time":     data.get("scan_time", ""),
        "total_scanned": data.get("total_scanned", 0),
        "pre_screened":  data.get("pre_screened", 0),
    }

    # 台股：附加籌碼排名資料
    if market == "tw":
        chip_map = _build_chip_map(date_str)
        if chip_map:
            for r in results:
                code = str(r.get("symbol", "")).replace(".TW", "").replace(".TWO", "")
                chip = chip_map.get(code)
                if chip:
                    r["chip_rank"]    = chip["rank"]
                    r["chip_net"]     = chip["total_net"]
                    r["chip_foreign"] = chip["foreign_net"]
                    r["chip_trust"]   = chip["trust_net"]

        # 附加融資融券資料
        margin_map = _build_margin_map(date_str)
        if margin_map:
            for r in results:
                code = str(r.get("symbol", "")).replace(".TW", "").replace(".TWO", "")
                margin = margin_map.get(code)
                if margin:
                    r["margin_data"] = margin

    # 附加歷史訊號脈絡
    history_map = _build_signal_history_map(market, date_str)
    for r in results:
        sym = r.get("symbol", "")
        r["signal_history"] = history_map.get(sym, [])

    # 附加歷史回測績效
    bt_stats = _load_backtest_stats(market)
    for r in results:
        sym = r.get("symbol", "")
        r["backtest_stats"] = bt_stats.get(sym)

    return {
        "buy":         [r for r in results if r.get("signal") == "BUY"],
        "watch":       [r for r in results if r.get("signal") == "WATCH"],
        "meta":        meta,
        "date_str":    date_str,
        "source_file": str(scan_file),
        "error":       None,
    }


# ── 結果儲存 ──────────────────────────────────────────────────────────────────

def save_claude_analysis(
    market: str,
    results: list,
    meta: dict,
    date_str: str = None,
) -> str:
    """
    將 Claude 分析結果儲存為 scan_results/claude_{market}_{date}.json。
    每個 result 需包含原始掃描欄位 + "claude_analysis" 字串。

    Returns: 輸出檔案路徑
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    output = {
        "market":               market,
        "scan_time":            meta.get("scan_time", ""),
        "total_scanned":        meta.get("total_scanned", 0),
        "pre_screened":         meta.get("pre_screened", 0),
        "claude_analyzed":      len(results),
        "claude_analysis_time": datetime.now().isoformat(),
        "results":              results,
    }

    out_file = SCAN_RESULTS_DIR / f"claude_{market}_{date_str}.json"
    atomic_write_json(out_file, output, ensure_ascii=False, indent=2, default=str)

    logger.info(f"已儲存 {len(results)} 支 Claude 分析 → {out_file}")

    # 補填進場時機到 scan_tracker（BUY 訊號才有進場時機評估）
    try:
        from scan_tracker import update_entry_timing
        timing_map = {}
        for r in results:
            if r.get("signal") != "BUY":
                continue
            analysis = r.get("claude_analysis") or ""
            idx = analysis.find("【明日開盤建議】")
            if idx == -1:
                idx = analysis.find("【進場時機評估】")  # 向後相容舊格式
            if idx == -1:
                continue
            section = analysis[idx: idx + 300]
            if "🟢" in section:
                timing_map[r["symbol"]] = "🟢"
            elif "🟡" in section:
                timing_map[r["symbol"]] = "🟡"
            elif "🔴" in section:
                timing_map[r["symbol"]] = "🔴"
        if timing_map:
            update_entry_timing(market, timing_map)
    except Exception as e:
        logger.warning(f"補填 entry_timing 失敗: {e}")

    # 補填完整 Claude 分析文字到 scan_tracker（BUY 訊號）
    try:
        from scan_tracker import update_entry_analysis
        analysis_map = {
            r["symbol"]: r["claude_analysis"]
            for r in results
            if r.get("signal") == "BUY" and r.get("claude_analysis")
        }
        if analysis_map:
            update_entry_analysis(market, analysis_map)
    except Exception as e:
        logger.warning(f"補填 claude_analysis 失敗: {e}")

    return str(out_file)


def get_latest_claude_results(market: str) -> dict | None:
    """讀取最新的 Claude 分析結果（供 FastAPI / 前端使用）"""
    files = sorted(SCAN_RESULTS_DIR.glob(f"claude_{market}_*.json"), reverse=True)
    if not files:
        return None
    try:
        with open(files[0], encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"讀取 Claude 結果失敗: {e}")
        return None


# ── 每日總結 ──────────────────────────────────────────────────────────────────

def build_daily_summary_prompt(date_str: str = None) -> str:
    """
    讀取台股 + 美股 Claude 分析結果，建立每日總結提示詞。
    由 Claude Code 排程任務呼叫（22:00），在個股分析完成後執行。
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    def _load_results(market: str) -> list:
        files = sorted(SCAN_RESULTS_DIR.glob(f"claude_{market}_{date_str}.json"), reverse=True)
        if not files:
            # fallback 最新一份
            files = sorted(SCAN_RESULTS_DIR.glob(f"claude_{market}_*.json"), reverse=True)
        if not files:
            return []
        try:
            with open(files[0], encoding="utf-8") as f:
                return json.load(f).get("results", [])
        except Exception:
            return []

    tw_results = _load_results("tw")
    us_results = _load_results("us")

    if not tw_results and not us_results:
        return ""

    def _fmt_stock_block(r: dict, market: str) -> str:
        sym     = r.get("symbol", "")
        name    = r.get("name", "")
        signal  = r.get("signal", "")
        close   = r.get("close", "")
        change  = r.get("change_pct", 0)
        vr      = r.get("volume_ratio", 1)
        sector  = r.get("sector", "")
        analysis = r.get("claude_analysis") or "（無分析）"
        # 只取前 600 字，避免 prompt 過長
        analysis_short = analysis[:600] + "..." if len(analysis) > 600 else analysis
        name_display = f"（{name}）" if name and name != sym else ""
        chip_info = ""
        if market == "tw" and r.get("chip_rank"):
            chip_info = f"　法人排名：第{r['chip_rank']}名"
        earnings_warn = "　⚠️財報近期" if r.get("earnings_soon") else ""
        return (
            f"[{signal}] {sym}{name_display}　板塊：{sector}\n"
            f"收盤：{close}　漲跌：{change:+.1f}%　量比：{vr:.1f}x{chip_info}{earnings_warn}\n"
            f"{analysis_short}\n"
        )

    tw_buy   = [r for r in tw_results if r.get("signal") == "BUY"]
    tw_watch = [r for r in tw_results if r.get("signal") == "WATCH"]
    us_buy   = [r for r in us_results if r.get("signal") == "BUY"]
    us_watch = [r for r in us_results if r.get("signal") == "WATCH"]

    sections = []

    if tw_buy or tw_watch:
        tw_lines = [f"### 台股（{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}）"]
        tw_lines.append(f"BUY {len(tw_buy)} 支　WATCH {len(tw_watch)} 支\n")
        for r in tw_buy:
            tw_lines.append(_fmt_stock_block(r, "tw"))
        for r in tw_watch:
            tw_lines.append(_fmt_stock_block(r, "tw"))
        sections.append("\n".join(tw_lines))

    if us_buy or us_watch:
        us_lines = [f"### 美股（{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}）"]
        us_lines.append(f"BUY {len(us_buy)} 支　WATCH {len(us_watch)} 支\n")
        for r in us_buy:
            us_lines.append(_fmt_stock_block(r, "us"))
        for r in us_watch:
            us_lines.append(_fmt_stock_block(r, "us"))
        sections.append("\n".join(us_lines))

    all_data = "\n\n".join(sections)

    return f"""你是資深跨市場技術分析師，今日已完成台股與美股所有 BUY/WATCH 訊號的個股分析。
請根據下方個股分析結果，撰寫一份今日盤後總結，協助快速掌握最值得關注的機會。

【今日掃描結果與個股分析】
{all_data}

請依以下格式輸出（繁體中文，語氣簡潔專業，每支股票說明不超過3行）：

【今日台股精選】
從台股 BUY/WATCH 中挑出 1~3 支最值得關注的，說明選擇原因（技術面優勢、籌碼配合、假突破風險低等）。
若無台股訊號則寫「今日台股無訊號」。

【今日美股精選】
從美股 BUY/WATCH 中挑出 1~3 支最值得關注的，說明選擇原因。
若無美股訊號則寫「今日美股無訊號」。

【需特別留意】
列出有財報近期警示、假突破風險高、或 Claude 建議🔴跳過的股票，並說明原因。
若無則寫「無特別警示」。

【今日市場觀察】
2~3 句話總結今日台美股整體訊號品質、板塊集中度或市場狀態，提供明日操作的大方向參考。"""


def save_daily_summary(summary_text: str, date_str: str = None) -> str:
    """儲存每日總結至 scan_results/daily_summary_YYYYMMDD.json"""
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    out = {
        "date":       date_str,
        "generated":  datetime.now().isoformat(),
        "summary":    summary_text,
    }
    out_file = SCAN_RESULTS_DIR / f"daily_summary_{date_str}.json"
    atomic_write_json(out_file, out, ensure_ascii=False, indent=2)
    logger.info(f"每日總結已儲存 → {out_file}")
    return str(out_file)


def get_latest_daily_summary() -> dict | None:
    """讀取最新每日總結（供 FastAPI / 前端使用）"""
    files = sorted(SCAN_RESULTS_DIR.glob("daily_summary_*.json"), reverse=True)
    if not files:
        return None
    try:
        with open(files[0], encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"讀取每日總結失敗: {e}")
        return None
