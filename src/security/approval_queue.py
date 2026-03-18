"""
Approval Queue
--------------
Manages request approval workflows for sensitive operations.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

try:
    from supabase import Client
except ImportError:
    Client = None


class ApprovalQueue:
    """Manages approval requests for sensitive operations."""

    def __init__(self, supabase_client: Optional[Client] = None):
        """
        Initialize approval queue.

        Args:
            supabase_client: Pre-initialized Supabase client (optional)
        """
        import os

        self.supabase = supabase_client
        self.use_supabase = supabase_client is not None
        self.fallback_file = Path("data/approval_requests.jsonl")

        if not self.use_supabase:
            self.fallback_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"[ApprovalQueue] Using JSON fallback at {self.fallback_file}")

    async def submit(
        self,
        action: str,
        username: str,
        justification: str,
    ) -> Optional[str]:
        """
        Submit a request for approval.

        Args:
            action: Action requiring approval (e.g., "export_billing")
            username: User requesting approval
            justification: Reason for the request

        Returns:
            Request ID if successful, None otherwise
        """
        try:
            record = {
                "action": action,
                "username": username,
                "justification": justification,
                "status": "pending",
                "approver": None,
                "created_at": datetime.utcnow().isoformat(),
                "resolved_at": None,
            }

            if self.use_supabase and self.supabase:
                result = self.supabase.table("approval_requests").insert(record).execute()
                if result.data:
                    return result.data[0].get("id")
            else:
                # Fallback: JSONL
                requests = self._load_fallback_requests()
                next_id = max([int(r.get("id", 0)) for r in requests], default=0) + 1
                record["id"] = str(next_id)
                requests.append(record)
                self._save_fallback_requests(requests)
                return str(next_id)

        except Exception as e:
            logger.error(f"[ApprovalQueue] Submit failed: {e}")

        return None

    async def approve(
        self,
        request_id: str,
        approver: str,
    ) -> bool:
        """
        Approve a pending request.

        Args:
            request_id: ID of the request
            approver: Username of approver

        Returns:
            True if successful
        """
        try:
            update_data = {
                "status": "approved",
                "approver": approver,
                "resolved_at": datetime.utcnow().isoformat(),
            }

            if self.use_supabase and self.supabase:
                self.supabase.table("approval_requests").update(update_data).eq(
                    "id", request_id
                ).execute()
                return True
            else:
                requests = self._load_fallback_requests()
                for r in requests:
                    if r.get("id") == request_id:
                        r.update(update_data)
                        self._save_fallback_requests(requests)
                        return True

        except Exception as e:
            logger.error(f"[ApprovalQueue] Approve failed: {e}")

        return False

    async def reject(
        self,
        request_id: str,
        approver: str,
        reason: str = "",
    ) -> bool:
        """
        Reject a pending request.

        Args:
            request_id: ID of the request
            approver: Username of approver
            reason: Reason for rejection

        Returns:
            True if successful
        """
        try:
            update_data = {
                "status": "rejected",
                "approver": approver,
                "resolved_at": datetime.utcnow().isoformat(),
            }

            if self.use_supabase and self.supabase:
                self.supabase.table("approval_requests").update(update_data).eq(
                    "id", request_id
                ).execute()
                return True
            else:
                requests = self._load_fallback_requests()
                for r in requests:
                    if r.get("id") == request_id:
                        r.update(update_data)
                        self._save_fallback_requests(requests)
                        return True

        except Exception as e:
            logger.error(f"[ApprovalQueue] Reject failed: {e}")

        return False

    async def check(self, request_id: str) -> Optional[dict]:
        """
        Check the status of a request.

        Args:
            request_id: ID of the request

        Returns:
            Request record if found
        """
        try:
            if self.use_supabase and self.supabase:
                result = self.supabase.table("approval_requests").select("*").eq(
                    "id", request_id
                ).execute()
                if result.data:
                    return result.data[0]
            else:
                requests = self._load_fallback_requests()
                for r in requests:
                    if r.get("id") == request_id:
                        return r

        except Exception as e:
            logger.error(f"[ApprovalQueue] Check failed: {e}")

        return None

    async def list_pending(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """
        List pending approval requests.

        Args:
            limit: Max results
            offset: Pagination offset

        Returns:
            List of pending requests
        """
        try:
            if self.use_supabase and self.supabase:
                result = self.supabase.table("approval_requests").select("*").eq(
                    "status", "pending"
                ).order("created_at", desc=True).limit(limit).offset(offset).execute()
                return result.data or []
            else:
                requests = self._load_fallback_requests()
                pending = [r for r in requests if r.get("status") == "pending"]
                pending.sort(key=lambda r: r.get("created_at", ""), reverse=True)
                return pending[offset : offset + limit]

        except Exception as e:
            logger.error(f"[ApprovalQueue] List pending failed: {e}")

        return []

    def _load_fallback_requests(self) -> list[dict]:
        """Load requests from JSONL fallback."""
        if not self.fallback_file.exists():
            return []

        requests = []
        try:
            with open(self.fallback_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        requests.append(json.loads(line))
        except Exception as e:
            logger.warning(f"[ApprovalQueue] Failed to load fallback: {e}")

        return requests

    def _save_fallback_requests(self, requests: list[dict]) -> None:
        """Save requests to JSONL fallback."""
        try:
            with open(self.fallback_file, "w") as f:
                for req in requests:
                    f.write(json.dumps(req) + "\n")
        except Exception as e:
            logger.warning(f"[ApprovalQueue] Failed to save fallback: {e}")
