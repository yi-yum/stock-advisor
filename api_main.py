from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import router_stock, router_btc, router_backtest, router_options, router_twse, router_watchlist, router_alerts, router_sector, router_history, router_crypto, router_backtest_stock, router_scanner, router_scan_history, router_sector_flow

app = FastAPI(
    title="Stock Advisor API",
    description="Stock and crypto analysis API",
    version="1.0.0"
)

# ── CORS 設定（開發階段允許所有 origin） ────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 包含所有 Router ────────────────────────────────
app.include_router(router_stock.router)
app.include_router(router_btc.router)
app.include_router(router_backtest.router)
app.include_router(router_options.router)
app.include_router(router_twse.router)
app.include_router(router_watchlist.router)
app.include_router(router_alerts.router)
app.include_router(router_sector.router)
app.include_router(router_history.router)
app.include_router(router_crypto.router)
app.include_router(router_backtest_stock.router)
app.include_router(router_scanner.router)
app.include_router(router_scan_history.router)
app.include_router(router_sector_flow.router)


# ── Startup Event ───────────────────────────────
@app.on_event("startup")
async def startup_event():
    import logging
    logging.basicConfig(level=logging.INFO)
    try:
        from scanner_scheduler import start_scheduler
        start_scheduler()
    except Exception as e:
        logging.getLogger("api_main").error(f"排程器啟動失敗: {e}")


# ── 健康檢查 endpoint ──────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# ── 根路徑 ────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Stock Advisor API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
