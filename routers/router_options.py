import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, HTTPException
from options_data import get_options_data, get_iv_percentile

router = APIRouter(prefix="/api", tags=["options"])


@router.get("/options/{symbol}")
async def get_options(symbol: str):
    """
    取得美股選擇權關鍵數據

    Args:
        symbol: 股票代號（如 'AAPL', 'TSLA'）

    Returns:
        JSON: 選擇權數據
    """
    try:
        result = get_options_data(symbol)

        # 檢查是否有錯誤
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取得選擇權數據失敗: {str(e)}")


@router.get("/options/{symbol}/iv")
async def get_options_iv(symbol: str):
    """
    取得選擇權隱含波動率

    Args:
        symbol: 股票代號

    Returns:
        JSON: IV 數據及分類
    """
    try:
        result = get_iv_percentile(symbol)

        # 檢查是否有錯誤
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"計算 IV 失敗: {str(e)}")
