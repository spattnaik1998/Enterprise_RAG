"""
Portal Database Layer
---------------------
All Supabase interactions for the client portal:
  - portal_users  : user lookup for authentication
  - service_tickets : ticket CRUD operations

Uses the service_role key (backend only -- never exposed to clients).
"""
from __future__ import annotations

import os
import secrets
import string
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Supabase client (lazy singleton)
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    global _client
    if _client is None:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_SERVICE_KEY", "")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
        _client = create_client(url, key)
    return _client


# ---------------------------------------------------------------------------
# portal_users helpers
# ---------------------------------------------------------------------------

def get_user_by_username(username: str) -> Optional[dict]:
    """Return the portal_users row for *username*, or None if not found."""
    try:
        resp = (
            _get_client()
            .table("portal_users")
            .select("*")
            .eq("username", username)
            .limit(1)
            .execute()
        )
        rows = resp.data
        return rows[0] if rows else None
    except Exception as exc:
        logger.error(f"[PortalDB] get_user_by_username error: {exc}")
        return None


def update_last_login(username: str) -> None:
    """Stamp last_login = now() for the given username (best-effort)."""
    try:
        (
            _get_client()
            .table("portal_users")
            .update({"last_login": datetime.now(timezone.utc).isoformat()})
            .eq("username", username)
            .execute()
        )
    except Exception as exc:
        logger.warning(f"[PortalDB] update_last_login failed (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# Ticket number generation
# ---------------------------------------------------------------------------

def _generate_ticket_number() -> str:
    """Return a unique ticket number like TKT-20250308-A3F9C1."""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    rand_part = "".join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    return f"TKT-{date_str}-{rand_part}"


# ---------------------------------------------------------------------------
# service_tickets helpers
# ---------------------------------------------------------------------------

def create_ticket(
    client_id: str,
    client_name: str,
    title: str,
    description: str,
    priority: str,
    category: str,
) -> dict:
    """Insert a new service ticket. Returns the created row."""
    ticket_number = _generate_ticket_number()
    row = {
        "ticket_number": ticket_number,
        "client_id": client_id,
        "client_name": client_name,
        "title": title,
        "description": description,
        "priority": priority,
        "category": category,
        "status": "open",
    }
    try:
        resp = (
            _get_client()
            .table("service_tickets")
            .insert(row)
            .execute()
        )
        created = resp.data[0] if resp.data else row
        logger.info(f"[PortalDB] Ticket created: {ticket_number} | client={client_id}")
        return created
    except Exception as exc:
        logger.error(f"[PortalDB] create_ticket error: {exc}")
        raise RuntimeError(f"Failed to create ticket: {exc}") from exc


def get_tickets_for_client(client_id: str) -> list[dict]:
    """Return all tickets for a specific client (newest first)."""
    try:
        resp = (
            _get_client()
            .table("service_tickets")
            .select("*")
            .eq("client_id", client_id)
            .order("created_at", desc=True)
            .execute()
        )
        return resp.data or []
    except Exception as exc:
        logger.error(f"[PortalDB] get_tickets_for_client error: {exc}")
        return []


def get_all_tickets(
    status_filter: Optional[str] = None,
    client_id_filter: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Return tickets for MSP view with optional filters (newest first)."""
    try:
        query = (
            _get_client()
            .table("service_tickets")
            .select("*")
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
        )
        if status_filter:
            query = query.eq("status", status_filter)
        if client_id_filter:
            query = query.eq("client_id", client_id_filter)
        resp = query.execute()
        return resp.data or []
    except Exception as exc:
        logger.error(f"[PortalDB] get_all_tickets error: {exc}")
        return []


def get_ticket_by_id(ticket_id: str) -> Optional[dict]:
    """Return a single ticket by UUID, or None."""
    try:
        resp = (
            _get_client()
            .table("service_tickets")
            .select("*")
            .eq("id", ticket_id)
            .limit(1)
            .execute()
        )
        rows = resp.data
        return rows[0] if rows else None
    except Exception as exc:
        logger.error(f"[PortalDB] get_ticket_by_id error: {exc}")
        return None


def update_ticket(ticket_id: str, updates: dict) -> Optional[dict]:
    """
    Update a ticket (MSP only). *updates* may include:
      status, assigned_to, engineer_notes, resolved_at
    Returns the updated row or None on error.
    """
    try:
        # Stamp resolved_at when moving to resolved/closed
        if updates.get("status") in ("resolved", "closed") and "resolved_at" not in updates:
            updates["resolved_at"] = datetime.now(timezone.utc).isoformat()
        resp = (
            _get_client()
            .table("service_tickets")
            .update(updates)
            .eq("id", ticket_id)
            .execute()
        )
        rows = resp.data
        updated = rows[0] if rows else None
        if updated:
            logger.info(f"[PortalDB] Ticket {ticket_id} updated: {updates}")
        return updated
    except Exception as exc:
        logger.error(f"[PortalDB] update_ticket error: {exc}")
        return None


def get_all_client_credentials() -> list[dict]:
    """
    Return all client portal users with credentials including passwords (for MSP distribution).
    """
    try:
        resp = (
            _get_client()
            .table("portal_users")
            .select("id,username,client_id,client_name,password_hash,created_at")
            .eq("role", "client")
            .order("client_name", desc=False)
            .execute()
        )
        return resp.data or []
    except Exception as exc:
        logger.error(f"[PortalDB] get_all_client_credentials error: {exc}")
        return []


def get_all_engineers() -> list[dict]:
    """Return all engineer profiles ordered by total_tickets desc."""
    try:
        resp = (
            _get_client()
            .table("engineers")
            .select("*")
            .order("total_tickets", desc=True)
            .execute()
        )
        return resp.data or []
    except Exception as exc:
        logger.error(f"[PortalDB] get_all_engineers error: {exc}")
        return []


def get_engineer_tickets(engineer_name: str) -> list[dict]:
    """Return all tickets assigned to an engineer.

    Handles both username (tom_okafor) and full name (Tom Okafor) formats
    for backward compatibility with existing tickets.
    """
    try:
        # Fetch tickets assigned to either username or full name
        # Username format: tom_okafor (from new dropdown)
        # Full name format: Tom Okafor (from initial backfill or legacy assignments)
        resp = (
            _get_client()
            .table("service_tickets")
            .select("*")
            .eq("assigned_to", engineer_name)
            .order("created_at", desc=True)
            .execute()
        )

        tickets = resp.data or []

        # If no tickets found with username, try with full name format
        # Convert username back to full name (e.g., tom_okafor -> Tom Okafor)
        if not tickets and "_" in engineer_name:
            full_name = engineer_name.replace("_", " ").title()
            resp2 = (
                _get_client()
                .table("service_tickets")
                .select("*")
                .eq("assigned_to", full_name)
                .order("created_at", desc=True)
                .execute()
            )
            tickets = resp2.data or []

        return tickets
    except Exception as exc:
        logger.error(f"[PortalDB] get_engineer_tickets error: {exc}")
        return []


def get_engineer_ticket_by_id(engineer_name: str, ticket_id: str) -> Optional[dict]:
    """Return a single ticket if it belongs to the engineer.

    Handles both username (tom_okafor) and full name (Tom Okafor) formats.
    """
    try:
        # Try to find ticket with username format
        resp = (
            _get_client()
            .table("service_tickets")
            .select("*")
            .eq("id", ticket_id)
            .eq("assigned_to", engineer_name)
            .limit(1)
            .execute()
        )
        rows = resp.data

        # If not found and engineer_name is a username, try with full name
        if not rows and "_" in engineer_name:
            full_name = engineer_name.replace("_", " ").title()
            resp = (
                _get_client()
                .table("service_tickets")
                .select("*")
                .eq("id", ticket_id)
                .eq("assigned_to", full_name)
                .limit(1)
                .execute()
            )
            rows = resp.data

        return rows[0] if rows else None
    except Exception as exc:
        logger.error(f"[PortalDB] get_engineer_ticket_by_id error: {exc}")
        return None


def update_engineer_ticket(
    engineer_name: str,
    ticket_id: str,
    updates: dict,
) -> Optional[dict]:
    """
    Update a ticket (engineer can only update their own assigned tickets).
    Returns updated row or None on error/unauthorized.
    """
    # Verify the ticket belongs to this engineer
    ticket = get_engineer_ticket_by_id(engineer_name, ticket_id)
    if ticket is None:
        logger.warning(f"[PortalDB] Engineer {engineer_name} tried to update ticket {ticket_id} they don't own")
        return None

    # Timestamp any status changes
    if updates.get("status") in ("resolved", "closed") and "resolved_at" not in updates:
        updates["resolved_at"] = datetime.now(timezone.utc).isoformat()

    try:
        resp = (
            _get_client()
            .table("service_tickets")
            .update(updates)
            .eq("id", ticket_id)
            .execute()
        )
        rows = resp.data
        updated = rows[0] if rows else None
        if updated:
            logger.info(f"[PortalDB] Engineer {engineer_name} updated ticket {ticket_id}: {updates}")
        return updated
    except Exception as exc:
        logger.error(f"[PortalDB] update_engineer_ticket error: {exc}")
        return None


def get_engineer_stats(engineer_name: str) -> dict:
    """Return ticket counts for an engineer grouped by status.

    Handles both username and full name formats for backward compatibility.
    """
    try:
        # Try with username format first
        resp = (
            _get_client()
            .table("service_tickets")
            .select("status")
            .eq("assigned_to", engineer_name)
            .execute()
        )
        rows = resp.data or []

        # If no results and engineer_name is a username, try with full name
        if not rows and "_" in engineer_name:
            full_name = engineer_name.replace("_", " ").title()
            resp = (
                _get_client()
                .table("service_tickets")
                .select("status")
                .eq("assigned_to", full_name)
                .execute()
            )
            rows = resp.data or []

        counts: dict[str, int] = {}
        for row in rows:
            s = row.get("status", "unknown")
            counts[s] = counts.get(s, 0) + 1
        return {
            "total": len(rows),
            "by_status": counts,
        }
    except Exception as exc:
        logger.error(f"[PortalDB] get_engineer_stats error: {exc}")
        return {"total": 0, "by_status": {}}


def get_ticket_stats() -> dict:
    """Return ticket counts grouped by status for the MSP dashboard."""
    try:
        resp = (
            _get_client()
            .table("service_tickets")
            .select("status")
            .execute()
        )
        rows = resp.data or []
        counts: dict[str, int] = {}
        for row in rows:
            s = row.get("status", "unknown")
            counts[s] = counts.get(s, 0) + 1
        return {
            "total": len(rows),
            "by_status": counts,
        }
    except Exception as exc:
        logger.error(f"[PortalDB] get_ticket_stats error: {exc}")
        return {"total": 0, "by_status": {}}
