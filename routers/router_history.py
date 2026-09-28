from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from history import load_history, delete_record

router = APIRouter(prefix="/api/history", tags=["history"])


class DeleteRequest(BaseModel):
    symbol: str
    timestamp: Optional[str] = None


@router.get("")
async def get_history():
    """Get all history records (max 100, newest first)."""
    try:
        records = load_history()
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{symbol}")
async def delete_history(symbol: str, timestamp: Optional[str] = None):
    """Delete history record(s) for a symbol."""
    try:
        delete_record(symbol, timestamp)
        return {"status": "ok", "symbol": symbol}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
