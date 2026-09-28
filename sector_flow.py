"""
sector_flow.py — Taiwan stock sector capital flow aggregator
Fetches TWSE institutional (三大法人) data and aggregates by sector.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

from json_utils import atomic_write_json

# Directory for daily snapshots
DATA_DIR = Path(__file__).parent / "sector_flow_data"
DATA_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)

INDUSTRY_MAP = {
    '01': '水泥工業', '02': '食品工業', '03': '塑膠工業', '04': '紡織纖維',
    '05': '電機機械', '06': '電器電纜', '07': '化學生技醫療', '08': '玻璃陶瓷',
    '09': '造紙工業', '10': '鋼鐵工業', '11': '橡膠工業', '12': '汽車工業',
    '13': '電子工業', '14': '建材營造', '15': '航運業', '16': '觀光餐旅',
    '17': '金融業', '18': '貿易百貨', '19': '綜合', '20': '其他',
    '21': '化學工業', '22': '生技醫療業', '23': '油電燃氣業', '24': '半導體業',
    '25': '電腦及週邊設備業', '26': '光電業', '27': '通信網路業', '28': '電子零組件業',
    '29': '電子通路業', '30': '資訊服務業', '31': '其他電子業', '32': '文化創意業',
    '33': '農業科技業', '34': '電子商務', '35': '綠能環保', '36': '數位雲端',
    '37': '運動休閒', '38': '居家生活',
}

# Simple in-memory cache: { date_str -> (timestamp, result) }
_cache: dict = {}
CACHE_TTL = 30 * 60  # 30 minutes


def _parse_int(s: str) -> int:
    """Parse a TWSE number string like '12,345' or '--' to int."""
    try:
        return int(str(s).replace(',', '').replace('+', '').strip())
    except (ValueError, AttributeError):
        return 0


def _fetch_sector_map() -> dict:
    """
    Fetch stock → {sector, name} mapping from TWSE OpenAPI.
    Returns {stock_code: {'sector': sector_name, 'name': short_name}}
    """
    url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch sector map: {e}")
        return {}

    mapping = {}
    for item in data:
        code = str(item.get('公司代號', '')).strip()
        industry_code = str(item.get('產業別', '')).strip().zfill(2)
        name = str(item.get('公司簡稱', '')).strip()
        if code and industry_code in INDUSTRY_MAP:
            mapping[code] = {'sector': INDUSTRY_MAP[industry_code], 'name': name}
    logger.info(f"Loaded sector map: {len(mapping)} stocks")
    return mapping


def _fetch_t86(date_str: str) -> list:
    """
    Fetch T86 institutional buy/sell data for a given date (YYYYMMDD).
    Returns list of dicts with keys: code, foreign_net, trust_net, dealer_net.
    Returns empty list if market was closed or data unavailable.
    """
    url = (
        f"https://www.twse.com.tw/rwd/zh/fund/T86"
        f"?date={date_str}&selectType=ALLBUT0999&response=json"
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch T86 for {date_str}: {e}")
        return []

    stat = payload.get('stat', '')
    if stat != 'OK':
        logger.info(f"T86 stat={stat!r} for {date_str} — market likely closed")
        return []

    raw_data = payload.get('data', [])
    if not raw_data:
        return []

    records = []
    for row in raw_data:
        if len(row) < 12:
            continue
        code = str(row[0]).strip()
        # 過濾 ETF（00xxx）、TDR（9字頭）、債券ETF（含英文字母後綴）
        if code.startswith('00') or code.startswith('9') or not code.isdigit():
            continue
        # 單位：股，除以 1000 轉換為張
        foreign_net = _parse_int(row[4])  // 1000   # 外陸資買賣超股數
        trust_net   = _parse_int(row[10]) // 1000   # 投信買賣超股數
        dealer_net  = _parse_int(row[11]) // 1000   # 自營商買賣超股數
        records.append({
            'code': code,
            'foreign_net': foreign_net,
            'trust_net': trust_net,
            'dealer_net': dealer_net,
        })

    logger.info(f"Fetched T86 for {date_str}: {len(records)} stocks")
    return records


def _build_result(date_str: str, sector_map: dict, t86_records: list) -> dict:
    """Aggregate T86 records by sector and return the result dict."""
    # sector_name -> aggregation bucket
    sectors: dict = {}

    for rec in t86_records:
        code = rec['code']
        info = sector_map.get(code)
        if info:
            sector_name = info['sector']
            stock_name  = info['name']
        else:
            sector_name = '其他'
            stock_name  = ''

        if sector_name not in sectors:
            sectors[sector_name] = {
                'name': sector_name,
                'foreign_net': 0,
                'trust_net': 0,
                'dealer_net': 0,
                'total_net': 0,
                'stock_count': 0,
                '_stocks': [],
            }

        b = sectors[sector_name]
        total = rec['foreign_net'] + rec['trust_net'] + rec['dealer_net']
        b['foreign_net'] += rec['foreign_net']
        b['trust_net']   += rec['trust_net']
        b['dealer_net']  += rec['dealer_net']
        b['total_net']   += total
        b['stock_count'] += 1
        b['_stocks'].append({
            'code': code,
            'name': stock_name,
            'foreign_net': rec['foreign_net'],
            'trust_net':   rec['trust_net'],
            'dealer_net':  rec['dealer_net'],
            'total_net':   total,
        })

    # Build sorted sector list
    sector_list = []
    for s in sectors.values():
        stocks_sorted = sorted(s['_stocks'], key=lambda x: x['total_net'], reverse=True)
        top = [st['code'] for st in sorted(s['_stocks'], key=lambda x: abs(x['foreign_net']), reverse=True)[:3]]
        sector_list.append({
            'name': s['name'],
            'foreign_net': s['foreign_net'],
            'trust_net': s['trust_net'],
            'dealer_net': s['dealer_net'],
            'total_net': s['total_net'],
            'stock_count': s['stock_count'],
            'top_stocks': top,
            'stocks': stocks_sorted,   # individual stock detail
        })

    sector_list.sort(key=lambda x: x['total_net'], reverse=True)

    total_foreign = sum(r['foreign_net'] for r in t86_records)
    total_trust   = sum(r['trust_net']   for r in t86_records)
    total_dealer  = sum(r['dealer_net']  for r in t86_records)

    return {
        'date': date_str,
        'sectors': sector_list,
        'total_foreign': total_foreign,
        'total_trust': total_trust,
        'total_dealer': total_dealer,
        'source': 'TWSE',
    }


def _prev_trading_day(date_str: str) -> str:
    """Return the previous calendar day (simple fallback, not holiday-aware)."""
    dt = datetime.strptime(date_str, '%Y%m%d') - timedelta(days=1)
    # Skip weekends
    while dt.weekday() >= 5:
        dt -= timedelta(days=1)
    return dt.strftime('%Y%m%d')


def get_tw_sector_flow(date_str: str = None) -> dict:
    """
    Main entry point.
    If date_str is None, try today then yesterday (skipping weekends).
    Results are cached for 30 minutes.
    """
    if date_str is None:
        today = datetime.now()
        # Skip weekends
        while today.weekday() >= 5:
            today -= timedelta(days=1)
        date_str = today.strftime('%Y%m%d')

    # Check cache (only valid if it has per-stock detail)
    if date_str in _cache:
        cached_ts, cached_result = _cache[date_str]
        sectors = cached_result.get('sectors', [])
        has_stocks = sectors and sectors[0].get('stocks')
        if time.time() - cached_ts < CACHE_TTL and has_stocks:
            logger.info(f"Returning cached sector flow for {date_str}")
            return cached_result

    logger.info(f"Fetching sector flow for {date_str}")

    sector_map = _fetch_sector_map()
    t86_records = _fetch_t86(date_str)

    # If no data (market closed), try the previous trading day
    if not t86_records:
        prev = _prev_trading_day(date_str)
        logger.info(f"No data for {date_str}, trying {prev}")
        t86_records = _fetch_t86(prev)
        if t86_records:
            date_str = prev

    result = _build_result(date_str, sector_map, t86_records)

    # Persist to disk (idempotent – overwrite is fine)
    _save_snapshot(date_str, result)

    # Store in cache
    _cache[date_str] = (time.time(), result)
    return result


# ── Persistence helpers ──────────────────────────────────────────────────────

def _snapshot_path(date_str: str) -> Path:
    return DATA_DIR / f"tw_{date_str}.json"


def _save_snapshot(date_str: str, result: dict) -> None:
    try:
        path = _snapshot_path(date_str)
        atomic_write_json(path, result, ensure_ascii=False, indent=2)
        logger.info(f"Saved sector flow snapshot: {path.name}")
    except Exception as e:
        logger.warning(f"Failed to save snapshot for {date_str}: {e}")


def _load_snapshot(date_str: str) -> dict | None:
    path = _snapshot_path(date_str)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        logger.warning(f"Failed to load snapshot {path.name}: {e}")
        return None


# ── History / accumulation ────────────────────────────────────────────────────

def _trading_days_before(end_date_str: str, count: int) -> list[str]:
    """Return up to `count` trading day strings (YYYYMMDD) ending at end_date_str (inclusive)."""
    dt = datetime.strptime(end_date_str, '%Y%m%d')
    days = []
    while len(days) < count:
        if dt.weekday() < 5:          # Mon–Fri only
            days.append(dt.strftime('%Y%m%d'))
        dt -= timedelta(days=1)
    return days  # newest first


def _compute_streak(daily_totals: list[int]) -> int:
    """
    Given a list of daily total_net values (newest first),
    return the consecutive buy (+N) or sell (-N) streak.
    e.g. [500, 300, 200, -100] → +3  (3 consecutive buy days)
         [-200, -100, 400] → -2
         [] or [0] → 0
    """
    if not daily_totals:
        return 0
    # Determine direction from most recent non-zero day
    sign = 0
    for v in daily_totals:
        if v > 0:
            sign = 1
            break
        if v < 0:
            sign = -1
            break
    if sign == 0:
        return 0
    streak = 0
    for v in daily_totals:
        if (sign > 0 and v > 0) or (sign < 0 and v < 0):
            streak += 1
        else:
            break
    return streak * sign


def _collect_snapshots(end_date_str: str, max_days: int, sector_map_ref: list) -> list[dict]:
    """
    Collect up to max_days snapshots ending at end_date_str (newest first).
    sector_map_ref is a 1-element list used as a mutable reference for lazy loading.
    """
    candidates = _trading_days_before(end_date_str, max_days * 2)
    collected: list[dict] = []
    for date_str in candidates:
        if len(collected) >= max_days:
            break
        snap = _load_snapshot(date_str)
        if snap and snap.get('sectors') and snap['sectors'] and snap['sectors'][0].get('stocks'):
            collected.append(snap)
            continue
        if sector_map_ref[0] is None:
            sector_map_ref[0] = _fetch_sector_map()
        records = _fetch_t86(date_str)
        if not records:
            continue
        result = _build_result(date_str, sector_map_ref[0], records)
        _save_snapshot(date_str, result)
        _cache[date_str] = (time.time(), result)
        collected.append(result)
    return collected  # newest first


def get_tw_sector_flow_history(days: int = 5, end_date_str: str = None) -> dict:
    """
    Return cumulative sector flow over the last `days` trading days.
    Also computes per-stock streak using up to 20 days of history.
    """
    if days < 1:
        days = 1
    if days > 60:
        days = 60

    if end_date_str is None:
        today = datetime.now()
        while today.weekday() >= 5:
            today -= timedelta(days=1)
        end_date_str = today.strftime('%Y%m%d')

    STREAK_WINDOW = max(days, 20)
    sector_map_ref = [None]

    # Collect extra days for streak (newest first)
    all_collected = _collect_snapshots(end_date_str, STREAK_WINDOW, sector_map_ref)

    if not all_collected:
        return {'dates': [], 'sectors': [], 'total_foreign': 0, 'total_trust': 0, 'total_dealer': 0, 'days': 0}

    # Use only the first `days` snapshots for accumulation
    for_acc = all_collected[:days]

    # Build streak lookup: code -> [daily total_net, ...] newest first
    stock_daily: dict[str, list[int]] = {}
    for snap in all_collected:
        for s in snap['sectors']:
            for st in s.get('stocks', []):
                code = st['code']
                if code not in stock_daily:
                    stock_daily[code] = []
                stock_daily[code].append(st['total_net'])

    # Aggregate accumulation period
    sector_acc: dict[str, dict] = {}
    stock_acc: dict[str, dict[str, dict]] = {}

    for snap in for_acc:
        for s in snap['sectors']:
            name = s['name']
            if name not in sector_acc:
                sector_acc[name] = {
                    'name': name,
                    'foreign_net': 0,
                    'trust_net': 0,
                    'dealer_net': 0,
                    'total_net': 0,
                    'stock_count': s['stock_count'],
                    'top_stocks': s['top_stocks'],
                }
                stock_acc[name] = {}
            b = sector_acc[name]
            b['foreign_net'] += s['foreign_net']
            b['trust_net']   += s['trust_net']
            b['dealer_net']  += s['dealer_net']
            b['total_net']   += s['total_net']

            for st in s.get('stocks', []):
                code = st['code']
                if code not in stock_acc[name]:
                    stock_acc[name][code] = {
                        'code': code,
                        'name': st['name'],
                        'foreign_net': 0,
                        'trust_net': 0,
                        'dealer_net': 0,
                        'total_net': 0,
                    }
                sa = stock_acc[name][code]
                sa['foreign_net'] += st['foreign_net']
                sa['trust_net']   += st['trust_net']
                sa['dealer_net']  += st['dealer_net']
                sa['total_net']   += st['total_net']

    for name, b in sector_acc.items():
        for st in stock_acc[name].values():
            st['streak'] = _compute_streak(stock_daily.get(st['code'], []))
        stocks_sorted = sorted(stock_acc[name].values(), key=lambda x: x['total_net'], reverse=True)
        b['stocks'] = stocks_sorted
        top = sorted(stock_acc[name].values(), key=lambda x: abs(x['foreign_net']), reverse=True)[:3]
        b['top_stocks'] = [s['code'] for s in top]

    sector_list = sorted(sector_acc.values(), key=lambda x: x['total_net'], reverse=True)

    dates_used = sorted([s['date'] for s in for_acc])
    return {
        'dates': dates_used,
        'date_from': dates_used[0] if dates_used else '',
        'date_to':   dates_used[-1] if dates_used else '',
        'days': len(for_acc),
        'sectors': sector_list,
        'total_foreign': sum(s['total_foreign'] for s in for_acc),
        'total_trust':   sum(s['total_trust']   for s in for_acc),
        'total_dealer':  sum(s['total_dealer']  for s in for_acc),
        'source': 'TWSE',
    }
