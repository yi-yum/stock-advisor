from fastapi import APIRouter, HTTPException
from pathlib import Path
import sys
import math
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from btc_short import get_btc_short_data

router = APIRouter(prefix="/api/btc", tags=["btc"])


def convert_nan_to_none(obj):
    """
    遞迴轉換 dict 中的 NaN 為 None，確保 JSON 序列化
    """
    if isinstance(obj, dict):
        return {k: convert_nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_none(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj):
            return None
        return obj
    else:
        return obj


@router.get("/short")
async def get_btc_short(coin: str = "BTC", anchor: str = "week", vwap_length: int = 14):
    """
    GET /api/btc/short?coin=BTC&anchor=week&vwap_length=14
    取得 BTC 短線交易數據
    """
    try:
        btc_data = get_btc_short_data(coin=coin, anchor=anchor, vwap_length=vwap_length)

        # 轉換 NaN 為 None
        btc_data = convert_nan_to_none(btc_data)

        return btc_data

    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})
