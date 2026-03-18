"""
Proactive Scheduler
------------------
APScheduler-based job scheduler for proactive alert generation.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, time

from loguru import logger

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
except ImportError:
    AsyncIOScheduler = None

from src.proactive.generators import (
    check_ar_risk,
    check_contract_expiry,
    check_quality_degradation,
)


class ProactiveScheduler:
    """Manages scheduled proactive alert jobs."""

    def __init__(self):
        """Initialize scheduler (not started until start() is called)."""
        if AsyncIOScheduler is None:
            logger.error("[Scheduler] APScheduler not installed; jobs disabled")
            self.scheduler = None
        else:
            self.scheduler = AsyncIOScheduler()

    async def start(self, alert_store) -> None:
        """
        Start the scheduler and register jobs.

        Args:
            alert_store: AlertStore instance for persisting alerts
        """
        if self.scheduler is None:
            logger.warning("[Scheduler] Cannot start; APScheduler not available")
            return

        try:
            # Job 1: AR risk check daily at 6 AM
            self.scheduler.add_job(
                self._run_ar_risk_check,
                "cron",
                hour=6,
                minute=0,
                args=[alert_store],
                id="check_ar_risk",
                name="Check AR Risk",
                replace_existing=True,
            )

            # Job 2: Contract expiry check daily at 6:30 AM
            self.scheduler.add_job(
                self._run_contract_check,
                "cron",
                hour=6,
                minute=30,
                args=[alert_store],
                id="check_contract_expiry",
                name="Check Contract Expiry",
                replace_existing=True,
            )

            # Job 3: Quality degradation check every 30 minutes
            self.scheduler.add_job(
                self._run_quality_check,
                "interval",
                minutes=30,
                args=[alert_store],
                id="check_quality_degradation",
                name="Check Quality Degradation",
                replace_existing=True,
            )

            self.scheduler.start()
            logger.info("[Scheduler] Started with 3 jobs (AR risk, contract expiry, quality checks)")

        except Exception as e:
            logger.error(f"[Scheduler] Failed to start: {e}")

    def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if self.scheduler is not None and self.scheduler.running:
            try:
                self.scheduler.shutdown(wait=False)
                logger.info("[Scheduler] Stopped")
            except Exception as e:
                logger.error(f"[Scheduler] Error stopping: {e}")

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self.scheduler is not None and self.scheduler.running

    @staticmethod
    async def _run_ar_risk_check(alert_store) -> None:
        """Run AR risk check job."""
        try:
            alerts = check_ar_risk()
            for alert in alerts:
                await alert_store.insert_alert(
                    alert_type=alert["alert_type"],
                    severity=alert["severity"],
                    payload=alert["payload"],
                )
            logger.info(f"[Job] AR risk check completed; {len(alerts)} alerts created")
        except Exception as e:
            logger.error(f"[Job] AR risk check failed: {e}")

    @staticmethod
    async def _run_contract_check(alert_store) -> None:
        """Run contract expiry check job."""
        try:
            alerts = check_contract_expiry()
            for alert in alerts:
                await alert_store.insert_alert(
                    alert_type=alert["alert_type"],
                    severity=alert["severity"],
                    payload=alert["payload"],
                )
            logger.info(f"[Job] Contract expiry check completed; {len(alerts)} alerts created")
        except Exception as e:
            logger.error(f"[Job] Contract expiry check failed: {e}")

    @staticmethod
    async def _run_quality_check(alert_store) -> None:
        """Run quality degradation check job."""
        try:
            alerts = check_quality_degradation()
            for alert in alerts:
                await alert_store.insert_alert(
                    alert_type=alert["alert_type"],
                    severity=alert["severity"],
                    payload=alert["payload"],
                )
            if alerts:
                logger.info(f"[Job] Quality check completed; {len(alerts)} alerts created")
        except Exception as e:
            logger.error(f"[Job] Quality check failed: {e}")
