import requests
from stock_data import safe_round

STOCKTWITS_API = "https://api.stocktwits.com/api/2"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
YF_SCREENER = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"


def fetch_most_active(count: int = 20) -> list[str]:
    """從 Yahoo Finance 抓即時成交量最大的股票代號"""
    try:
        r = requests.get(
            YF_SCREENER,
            params={"formatted": "true", "lang": "en-US", "region": "US",
                    "scrIds": "most_actives", "count": count},
            headers=HEADERS,
            timeout=8,
        )
        if r.status_code != 200:
            return []
        quotes = r.json().get("finance", {}).get("result", [{}])[0].get("quotes", [])
        return [q["symbol"] for q in quotes if q.get("symbol")]
    except Exception:
        return []


def fetch_posts(symbol: str, limit: int = 30) -> list:
    """抓取特定股票的最新貼文"""
    try:
        r = requests.get(
            f"{STOCKTWITS_API}/streams/symbol/{symbol}.json",
            params={"limit": limit},
            headers=HEADERS,
            timeout=8,
        )
        if r.status_code != 200:
            return []

        messages = r.json().get("messages", [])
        result = []
        for m in messages:
            sentiment = None
            if m.get("entities", {}).get("sentiment"):
                sentiment = m["entities"]["sentiment"].get("basic")

            result.append({
                "id":        m.get("id"),
                "body":      m.get("body", ""),
                "created":   m.get("created_at", "")[:16].replace("T", " "),
                "user":      m.get("user", {}).get("username", ""),
                "sentiment": sentiment,
                "likes":     m.get("likes", {}).get("total", 0),
            })
        return result
    except Exception:
        return []


def fetch_symbol_with_sentiment(symbol: str) -> dict | None:
    """抓取一支股票的貼文並計算情緒，用於批次掃描"""
    posts = fetch_posts(symbol, limit=20)
    if not posts:
        return None
    summary = summarize_sentiment(posts)
    summary["symbol"] = symbol
    summary["posts"]  = posts
    return summary


def scan_watchlist(symbols: list) -> list:
    """批次掃描清單，回傳依討論量排序的結果"""
    results = []
    for symbol in symbols:
        data = fetch_symbol_with_sentiment(symbol)
        if data:
            results.append(data)
    results.sort(key=lambda x: x["total"], reverse=True)
    return results


def summarize_sentiment(posts: list) -> dict:
    """統計貼文情緒比例"""
    tagged   = [p for p in posts if p["sentiment"]]
    bullish  = sum(1 for p in tagged if p["sentiment"] == "Bullish")
    bearish  = sum(1 for p in tagged if p["sentiment"] == "Bearish")
    total    = len(posts)
    tagged_n = len(tagged)

    if tagged_n == 0:
        mood = "無標記"
    elif bullish / tagged_n >= 0.65:
        mood = "偏多"
    elif bearish / tagged_n >= 0.65:
        mood = "偏空"
    else:
        mood = "分歧"

    return {
        "total":       total,
        "tagged":      tagged_n,
        "bullish":     bullish,
        "bearish":     bearish,
        "bullish_pct": safe_round(bullish / tagged_n * 100, 1) if tagged_n else None,
        "bearish_pct": safe_round(bearish / tagged_n * 100, 1) if tagged_n else None,
        "mood":        mood,
    }


