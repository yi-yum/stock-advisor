import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, HTTPException
from twse_data import get_twse_institutional, get_twse_margin
from twse_realtime import get_multiple_twse_realtime, is_tw_market_open

router = APIRouter(prefix="/api", tags=["twse"])


@router.get("/twse/institutional/{symbol}")
async def get_institutional(symbol: str):
    """
    取得台股三大法人買賣超數據

    Args:
        symbol: 股票代號（支援 "2330", "2330.TW" 格式）

    Returns:
        JSON: 三大法人買賣超數據
    """
    try:
        result = get_twse_institutional(symbol)

        # 檢查是否有錯誤
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取得機構買賣超失敗: {str(e)}")


@router.get("/twse/margin/{symbol}")
async def get_margin(symbol: str):
    """
    取得台股融資融券數據

    Args:
        symbol: 股票代號（支援 "2330", "2330.TW" 格式）

    Returns:
        JSON: 融資融券數據
    """
    try:
        result = get_twse_margin(symbol)

        # 檢查是否有錯誤
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取得融資融券失敗: {str(e)}")


@router.get("/twse/realtime")
async def get_realtime(symbols: str):
    """
    取得多支台股即時報價

    Args:
        symbols: 逗號分隔的股票代號字串（如 "2330.TW,2317.TW"）

    Returns:
        JSON: 多支股票即時報價
    """
    try:
        # 解析逗號分隔的符號
        if not symbols or not symbols.strip():
            raise HTTPException(status_code=400, detail="symbols 參數不能為空")

        symbols_list = [s.strip() for s in symbols.split(",") if s.strip()]

        if not symbols_list:
            raise HTTPException(status_code=400, detail="無效的 symbols 參數")

        # 取得即時報價
        result = get_multiple_twse_realtime(symbols_list)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"取得即時報價失敗: {str(e)}")


@router.get("/twse/market-status")
async def get_market_status():
    """
    檢查台股市場是否開盤

    Returns:
        JSON: {"is_open": bool}
    """
    try:
        is_open = is_tw_market_open()
        return {"is_open": is_open}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"檢查市場狀態失敗: {str(e)}")
