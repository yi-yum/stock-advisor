import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api", tags=["sector-flow"])
logger = logging.getLogger(__name__)


@router.get("/sector-flow/tw")
async def get_tw_sector_flow_api(date: str = None):
    """
    Get Taiwan stock sector capital flow (三大法人買賣超 by sector).
    Optional query param: date (YYYYMMDD). Defaults to latest trading day.
    """
    try:
        from sector_flow import get_tw_sector_flow
        return get_tw_sector_flow(date)
    except Exception as e:
        logger.error(f"sector-flow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sector-flow/tw/history")
async def get_tw_sector_flow_history_api(
    days: int = Query(default=5, ge=1, le=60, description="Number of trading days to accumulate"),
    end_date: str = Query(default=None, description="End date YYYYMMDD, defaults to latest trading day"),
):
    """
    Get accumulated sector capital flow over the last N trading days.
    """
    try:
        from sector_flow import get_tw_sector_flow_history
        return get_tw_sector_flow_history(days=days, end_date_str=end_date)
    except Exception as e:
        logger.error(f"sector-flow history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sector-flow/us")
async def get_us_sector_flow_api(
    days: int = Query(default=20, ge=1, le=60, description="Number of trading days for performance calculation"),
):
    """
    Get US sector rotation data via SPDR sector ETFs (XLK, XLF, XLE, etc.) vs SPY.
    """
    try:
        from us_sector_flow import get_us_sector_flow
        return get_us_sector_flow(days=days)
    except Exception as e:
        logger.error(f"us sector-flow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
