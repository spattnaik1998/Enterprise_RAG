"""
Alert Store
----------
Persists and retrieves proactive alerts (Supabase backed with JSON fallback).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

try:
    from supabase import Client, create_client
except ImportError:
    Client = None
    create_client = None


class AlertStore:
    """Manages proactive alerts (CRUD operations)."""

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        supabase_client: Optional[Client] = None,
    ):
        """
        Initialize alert store.

        Args:
            supabase_url: Supabase project URL (falls back to env var)
            supabase_key: Supabase service key (falls back to env var)
            supabase_client: Pre-initialized Supabase client (takes precedence)
        """
        import os

        self.supabase: Optional[Client] = None
        self.use_supabase = False
        self.fallback_file = Path("data/proactive_alerts.jsonl")

        # Use provided client if available
        if supabase_client is not None:
            self.supabase = supabase_client
            self.use_supabase = True
            logger.info("[AlertStore] Using provided Supabase client")
        else:
            # Otherwise try to create from URL/key
            supabase_url = supabase_url or os.getenv("SUPABASE_URL")
            supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_KEY")

            if supabase_url and supabase_key and Client is not None:
                try:
                    self.supabase = create_client(supabase_url, supabase_key)
                    self.use_supabase = True
                    logger.info("[AlertStore] Using Supabase backend")
                except Exception as e:
                    logger.warning(f"[AlertStore] Supabase initialization failed: {e}; using JSON fallback")

        if not self.use_supabase:
            self.fallback_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"[AlertStore] Using JSON fallback at {self.fallback_file}")

    async def insert_alert(self, alert_type: str, severity: str, payload: dict) -> Optional[str]:
        """
        Insert a new alert.

        Args:
            alert_type: "ar_risk", "contract_expiry", "quality_degradation"
            severity: "high", "medium", "low"
            payload: dict with alert details

        Returns:
            Alert ID if successful, None otherwise
        """
        try:
            record = {
                "alert_type": alert_type,
                "severity": severity,
                "payload": payload,
                "acknowledged": False,
                "created_at": datetime.utcnow().isoformat(),
            }

            if self.use_supabase:
                result = self.supabase.table("proactive_alerts").insert(record).execute()
                if result.data:
                    return result.data[0].get("id")
            else:
                # Fallback: append to JSONL with auto-incremented ID
                alerts = self._load_fallback_alerts()
                next_id = max([int(a.get("id", 0)) for a in alerts], default=0) + 1
                record["id"] = str(next_id)
                alerts.append(record)
                self._save_fallback_alerts(alerts)
                return str(next_id)

        except Exception as e:
            logger.error(f"[AlertStore] Insert failed: {e}")

        return None

    async def list_alerts(
        self,
        limit: int = 50,
        offset: int = 0,
        acknowledged: Optional[bool] = None,
    ) -> list[dict]:
        """
        List alerts with optional filtering.

        Args:
            limit: Max number of alerts
            offset: Pagination offset
            acknowledged: Filter by acknowledgment status (None = all)

        Returns:
            List of alert dicts
        """
        try:
            if self.use_supabase:
                query = self.supabase.table("proactive_alerts").select("*")
                if acknowledged is not None:
                    query = query.eq("acknowledged", acknowledged)
                query = query.order("created_at", desc=True)
                query = query.limit(limit).offset(offset)
                result = query.execute()
                return result.data or []
            else:
                alerts = self._load_fallback_alerts()
                if acknowledged is not None:
                    alerts = [a for a in alerts if a.get("acknowledged") == acknowledged]
                alerts.sort(key=lambda a: a.get("created_at", ""), reverse=True)
                return alerts[offset : offset + limit]

        except Exception as e:
            logger.error(f"[AlertStore] List failed: {e}")

        return []

    async def get_alert(self, alert_id: str) -> Optional[dict]:
        """Get a single alert by ID."""
        try:
            if self.use_supabase:
                result = self.supabase.table("proactive_alerts").select("*").eq("id", alert_id).execute()
                if result.data:
                    return result.data[0]
            else:
                alerts = self._load_fallback_alerts()
                for a in alerts:
                    if a.get("id") == alert_id:
                        return a

        except Exception as e:
            logger.error(f"[AlertStore] Get failed: {e}")

        return None

    async def acknowledge_alert(self, alert_id: str, acknowledged: bool = True) -> bool:
        """Mark an alert as acknowledged."""
        try:
            if self.use_supabase:
                self.supabase.table("proactive_alerts").update(
                    {"acknowledged": acknowledged}
                ).eq("id", alert_id).execute()
                return True
            else:
                alerts = self._load_fallback_alerts()
                for a in alerts:
                    if a.get("id") == alert_id:
                        a["acknowledged"] = acknowledged
                        self._save_fallback_alerts(alerts)
                        return True

        except Exception as e:
            logger.error(f"[AlertStore] Acknowledge failed: {e}")

        return False

    def _load_fallback_alerts(self) -> list[dict]:
        """Load alerts from JSON fallback file."""
        if not self.fallback_file.exists():
            return []

        alerts = []
        try:
            with open(self.fallback_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        alerts.append(json.loads(line))
        except Exception as e:
            logger.warning(f"[AlertStore] Failed to load fallback: {e}")

        return alerts

    def _save_fallback_alerts(self, alerts: list[dict]) -> None:
        """Save alerts to JSON fallback file."""
        try:
            with open(self.fallback_file, "w") as f:
                for alert in alerts:
                    f.write(json.dumps(alert) + "\n")
        except Exception as e:
            logger.warning(f"[AlertStore] Failed to save fallback: {e}")
