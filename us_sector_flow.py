"""
us_sector_flow.py — US sector rotation tracker via SPDR sector ETFs
Uses yfinance to fetch 11 sector ETFs and SPY, computes N-day performance
and relative strength vs SPY as a proxy for capital flow rotation.
"""

import logging
import time
from datetime import datetime, timedelta

import yfinance as yf

logger = logging.getLogger(__name__)

# SPDR sector ETFs with Chinese names
SECTOR_ETFS: dict[str, str] = {
    'XLK':  '科技',
    'XLF':  '金融',
    'XLE':  '能源',
    'XLV':  '醫療保健',
    'XLI':  '工業',
    'XLY':  '非必需消費',
    'XLP':  '必需消費',
    'XLU':  '公用事業',
    'XLRE': '房地產',
    'XLC':  '通信服務',
    'XLB':  '原材料',
}

# In-memory cache: { days -> (timestamp, result) }
_cache: dict = {}
CACHE_TTL = 30 * 60  # 30 minutes


def get_us_sector_flow(days: int = 20) -> dict:
    """
    Return US sector rotation data based on SPDR ETF performance.

    Response fields per sector:
      name         Chinese sector name
      ticker       ETF ticker (e.g. XLK)
      change_pct   N-day price return (%)
      vs_spy       Excess return vs SPY (%)
      volume_ratio Recent 5-day avg volume / 20-day avg volume
      close        Latest close price
      trend        'up' | 'down' | 'flat'
    """
    if days < 1:
        days = 1
    if days > 60:
        days = 60

    cache_key = days
    if cache_key in _cache:
        ts, result = _cache[cache_key]
        if time.time() - ts < CACHE_TTL:
            logger.info(f"Returning cached US sector flow (days={days})")
            return result

    # Fetch extra buffer so we always have enough trading days
    fetch_period = f"{days + 30}d"
    symbols = ['SPY'] + list(SECTOR_ETFS.keys())

    try:
        raw = yf.download(
            symbols,
            period=fetch_period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        close_df = raw['Close'].dropna(how='all')
        volume_df = raw['Volume'].dropna(how='all')
    except Exception as e:
        logger.error(f"yfinance download failed: {e}")
        return _empty_response(days)

    if close_df.empty or len(close_df) < 2:
        return _empty_response(days)

    # Slice to last `days` trading days (+ 1 for the base price)
    close_df = close_df.tail(days + 1)
    volume_df = volume_df.tail(days + 20)   # need more for vol ratio

    spy_start = close_df['SPY'].iloc[0]
    spy_end   = close_df['SPY'].iloc[-1]
    spy_change = (spy_end / spy_start - 1) * 100 if spy_start else 0.0

    date_from = close_df.index[0].strftime('%Y-%m-%d')
    date_to   = close_df.index[-1].strftime('%Y-%m-%d')

    sectors = []
    for ticker, name in SECTOR_ETFS.items():
        if ticker not in close_df.columns:
            continue

        col_close  = close_df[ticker].dropna()
        col_volume = volume_df[ticker].dropna() if ticker in volume_df.columns else None

        if len(col_close) < 2:
            continue

        start = col_close.iloc[0]
        end   = col_close.iloc[-1]
        change_pct = (end / start - 1) * 100 if start else 0.0
        vs_spy     = change_pct - spy_change

        # Volume ratio: recent 5-day avg vs prior 15-day avg
        vol_ratio = 1.0
        if col_volume is not None and len(col_volume) >= 10:
            recent = col_volume.iloc[-5:].mean()
            prior  = col_volume.iloc[-20:-5].mean() if len(col_volume) >= 20 else col_volume.iloc[:-5].mean()
            vol_ratio = round(float(recent / prior), 2) if prior > 0 else 1.0

        # Simple trend determination
        if vs_spy > 0.5:
            trend = 'up'
        elif vs_spy < -0.5:
            trend = 'down'
        else:
            trend = 'flat'

        sectors.append({
            'name':         name,
            'ticker':       ticker,
            'change_pct':   round(float(change_pct), 2),
            'vs_spy':       round(float(vs_spy), 2),
            'volume_ratio': vol_ratio,
            'close':        round(float(end), 2),
            'trend':        trend,
        })

    # Sort by vs_spy descending (best performing vs market first)
    sectors.sort(key=lambda x: x['vs_spy'], reverse=True)

    result = {
        'market':       'us',
        'period_days':  days,
        'date_from':    date_from,
        'date_to':      date_to,
        'spy_change_pct': round(float(spy_change), 2),
        'sectors':      sectors,
        'source':       'SPDR ETF / Yahoo Finance',
    }

    _cache[cache_key] = (time.time(), result)
    return result


def _empty_response(days: int) -> dict:
    return {
        'market': 'us',
        'period_days': days,
        'date_from': '',
        'date_to': '',
        'spy_change_pct': 0.0,
        'sectors': [],
        'source': 'SPDR ETF / Yahoo Finance',
    }
