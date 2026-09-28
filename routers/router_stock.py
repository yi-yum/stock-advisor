from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from stock_data import get_stock_data
from stocktwits import fetch_symbol_with_sentiment

router = APIRouter(prefix="/api/stock", tags=["stock"])


class SentimentRequest(BaseModel):
    symbol: str


@router.get("/{symbol}")
async def get_stock(symbol: str):
    """
    GET /api/stock/{symbol}
    回傳股票摘要資料
    """
    try:
        summary_dict, df, error = get_stock_data(symbol)

        if error is not None:
            raise HTTPException(status_code=400, detail={"error": error})

        if summary_dict is None:
            raise HTTPException(status_code=400, detail={"error": f"無法取得 {symbol} 數據"})

        return summary_dict

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.post("/sentiment")
async def get_sentiment(request: SentimentRequest):
    """
    POST /api/stock/sentiment
    從 StockTwits 取得社群情緒
    """
    try:
        symbol = request.symbol
        result = fetch_symbol_with_sentiment(symbol)

        if result is None:
            return {"error": f"無法取得 {symbol} 的社群情緒數據"}

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})
