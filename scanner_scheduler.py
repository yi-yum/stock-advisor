import logging
import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger("scanner_scheduler")
TW_TZ = pytz.timezone("Asia/Taipei")
scheduler = AsyncIOScheduler()


def start_scheduler():
    """由 api_main.py 的 startup event 呼叫"""
    from scanner import run_tw_scan, run_us_scan

    async def _tw_job():
        try:
            logger.info("排程觸發：台股掃描開始")
            result = await run_tw_scan()
            logger.info(f"台股掃描完成：{result.get('analyzed', 0)} 支訊號")
        except Exception as e:
            logger.error(f"台股排程掃描失敗: {e}", exc_info=True)

    async def _us_job():
        try:
            logger.info("排程觸發：美股掃描開始")
            result = await run_us_scan()
            logger.info(f"美股掃描完成：{result.get('analyzed', 0)} 支訊號")
        except Exception as e:
            logger.error(f"美股排程掃描失敗: {e}", exc_info=True)

    scheduler.add_job(
        _tw_job,
        CronTrigger(hour=15, minute=0, day_of_week="mon-fri", timezone=TW_TZ),
        id="tw_daily_scan", replace_existing=True,
    )
    scheduler.add_job(
        _us_job,
        CronTrigger(hour=21, minute=0, day_of_week="mon-fri", timezone=TW_TZ),
        id="us_daily_scan", replace_existing=True,
    )

    # ── 加密貨幣訊號檢查（每 4 小時，全週 24/7）──────────────────────────────
    async def _crypto_notify_job():
        try:
            logger.info("排程觸發：加密貨幣訊號檢查")
            from routers.router_crypto import check_and_notify
            result = check_and_notify()
            n = result.get("alerts", 0)
            logger.info(f"加密貨幣訊號檢查完成：{n} 個新訊號" + (
                f"，LINE 已發送 {len(result.get('line_results', []))} 則" if n else ""
            ))
        except Exception as e:
            logger.error(f"加密貨幣訊號檢查失敗: {e}", exc_info=True)

    from apscheduler.triggers.interval import IntervalTrigger
    scheduler.add_job(
        _crypto_notify_job,
        IntervalTrigger(hours=1),
        id="crypto_notify", replace_existing=True,
    )

    if not scheduler.running:
        scheduler.start()
    logger.info("Scanner scheduler 已啟動（台股 15:00 / 美股 21:00，週一到週五；加密貨幣每 1 小時）")
