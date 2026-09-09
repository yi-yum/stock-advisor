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


def build_st_gemini_message(symbol: str, posts: list, summary: dict) -> str:
    """組裝給 Gemini 的社群情緒分析 prompt"""
    sentiment_line = (
        f"看多 {summary['bullish_pct']}% / 看空 {summary['bearish_pct']}%"
        if summary["tagged"] else "無標記數據"
    )
    posts_text = "\n".join([
        f"- [{p['created']}] @{p['user']} "
        f"{'[看多]' if p['sentiment']=='Bullish' else '[看空]' if p['sentiment']=='Bearish' else ''} "
        f"{p['body'][:120]}"
        for p in posts[:15]
    ])

    return f"""以下是 StockTwits 上 ${symbol} 的近期社群討論（共 {summary['total']} 筆）：

情緒統計：{sentiment_line}（整體氛圍：{summary['mood']}）

貼文內容：
{posts_text}

請分析：
1. 社群整體情緒傾向（看多/看空/分歧）
2. 貼文中反覆出現的關鍵主題（財報、產品、法規、市場事件等）
3. 是否有值得注意的特殊訊息（重大消息、異常情緒）
4. 社群情緒對短中期股價的潛在影響

請用繁體中文回答，簡潔扼要。"""


def build_trending_gemini_message(trending_data: list) -> str:
    """組裝熱門股票總覽給 Gemini 分析"""
    lines = []
    for d in trending_data:
        sentiment_str = (
            f"看多 {d['bullish_pct']}% / 看空 {d['bearish_pct']}%"
            if d.get("bullish_pct") is not None else "無標記"
        )
        # 取前3筆貼文摘要
        snippets = "；".join([p["body"][:60] for p in d.get("posts", [])[:3]])
        lines.append(f"- ${d['symbol']}：{d['total']} 筆討論，{sentiment_str}，近期內容：{snippets}")

    return f"""以下是 StockTwits 今日熱門討論股票（共 {len(trending_data)} 支）：

{chr(10).join(lines)}

請分析：
1. 哪幾支股票的討論最值得關注？原因為何？
2. 各股票的社群情緒是否反映出特定的市場主題或事件？
3. 有哪些股票出現異常高熱度或極端情緒，可能需要特別留意？
4. 整體來看，目前市場社群的焦點在哪個板塊或主題？

請用繁體中文回答，條列式輸出。"""
