import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import yfinance as yf
from alerts import load_alerts, add_alert, remove_alert, check_alerts

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


class AddAlertRequest(BaseModel):
    symbol: str
    alert_type: str
    price: float
    note: str = ""


class CheckAlertsRequest(BaseModel):
    symbols: Optional[List[str]] = None


@router.get("")
def get_alerts():
    """Get all alerts"""
    try:
        alerts = load_alerts()
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("")
def add_new_alert(request: AddAlertRequest):
    """Add a new price alert"""
    try:
        success, message = add_alert(request.symbol, request.alert_type, request.price, request.note)
        return {"success": success, "message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{symbol}/{price}")
def delete_alert(symbol: str, price: float):
    """Remove a price alert"""
    try:
        success, message = remove_alert(symbol, price)
        return {"success": success, "message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check")
def check_price_alerts(request: CheckAlertsRequest):
    """Check if any alerts have been triggered"""
    try:
        # If no symbols specified, check all alerts
        if not request.symbols:
            alerts = load_alerts()
            symbols = list(set([a["symbol"] for a in alerts]))
        else:
            symbols = request.symbols

        # Fetch current prices using yfinance
        current_prices = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                price = ticker.fast_info.get("last_price")
                if price is not None:
                    current_prices[symbol] = price
            except Exception:
                pass  # Skip symbols that fail to fetch

        # Check alerts
        triggered = check_alerts(current_prices)

        # Get all alerts for comparison
        all_alerts = load_alerts()

        return {
            "triggered": triggered,
            "all_alerts": all_alerts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
