import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger("router_scan_history")
router = APIRouter(prefix="/api/scan-history", tags=["scan_history"])

SCAN_RESULTS_DIR = Path(__file__).parent.parent / "scan_results"


@router.get("")
def get_scan_history(limit: int = Query(30, description="最多回傳幾筆")):
    """
    回傳歷史掃描紀錄清單（每次掃描的摘要，不含完整結果）。
    按日期降序排列。
    """
    try:
        files = sorted(SCAN_RESULTS_DIR.glob("*.json"), reverse=True)[:limit]
        history = []
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                results = data.get("results", [])
                buy_signals = [r for r in results if r.get("signal") == "BUY" or
                               (r.get("gemini_analysis") or {}).get("direction") == "BUY"]
                history.append({
                    "filename":      f.name,
                    "market":        data.get("market", ""),
                    "scan_time":     data.get("scan_time", ""),
                    "total_scanned": data.get("total_scanned", 0),
                    "pre_screened":  data.get("pre_screened", 0),
                    "analyzed":      data.get("analyzed", 0),
                    "buy_count":     len(buy_signals),
                    "top_signals":   buy_signals[:5],  # 只回傳前 5 個供預覽
                })
            except Exception as e:
                logger.warning(f"讀取 {f.name} 失敗: {e}")
        return {"history": history, "total": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/detail")
def get_scan_detail(filename: str = Query(..., description="檔案名稱，如 us_20260914.json")):
    """回傳單次掃描的完整結果。"""
    # 安全性：只允許 {tw|us}_{date}.json 格式
    import re
    if not re.match(r'^(tw|us)_\d{8}\.json$', filename):
        raise HTTPException(status_code=400, detail="無效的檔案名稱格式")
    filepath = SCAN_RESULTS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="找不到該掃描紀錄")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
