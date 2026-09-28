import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, HTTPException
from sector_scan import scan_sectors

router = APIRouter(prefix="/api/sector", tags=["sector"])


@router.get("/scan")
def scan_sector():
    """Scan sectors for opportunities (this is a slow operation, ~30 seconds)"""
    try:
        sectors = scan_sectors()
        return {"sectors": sectors}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
