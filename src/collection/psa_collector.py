"""
PSA Collector (ConnectWise Manage simulation)
----------------------------------------------
Reads psa_tickets.json from data/enterprise/ and converts each service
ticket into a rich-text RawDocument for RAG indexing.

Each document captures:
  - Ticket header (ID, type, priority, status)
  - SLA compliance
  - Technician assignment and hours billed
  - Billing status and resolution notes
  - Client relationship context
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from loguru import logger

from src.collection.base_collector import BaseCollector
from src.schemas import RawDocument, SourceType


class PSACollector(BaseCollector):
    """
    Yields one RawDocument per service ticket from the ConnectWise-style PSA export.

    Tickets are represented as natural-language narratives so that queries like
    "which critical security tickets breached SLA?" work through semantic search.
    """

    source_type = SourceType.PSA

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.data_file = Path(config.get("data_file", "data/enterprise/psa_tickets.json"))

    async def health_check(self) -> bool:
        ok = self.data_file.exists()
        if not ok:
            logger.warning(f"[PSA] Data file not found: {self.data_file}")
        return ok

    async def collect(self) -> AsyncIterator[RawDocument]:
        logger.info(f"[PSA] Loading tickets from {self.data_file}")
        try:
            raw = json.loads(self.data_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error(f"[PSA] Failed to read data file: {exc}")
            return

        tickets = raw if isinstance(raw, list) else raw.get("records", [])
        logger.info(f"[PSA] Processing {len(tickets)} ticket records")

        for ticket in tickets:
            try:
                yield self._to_document(ticket)
            except Exception as exc:
                logger.warning(f"[PSA] Skipping ticket {ticket.get('ticket_id', '?')}: {exc}")

    # --- Document construction ------------------------------------------------

    def _to_document(self, t: dict) -> RawDocument:
        # Use actual field names from the data
        tid         = t["ticket_id"]
        client_name = t["client_name"]
        status      = t["status"]
        priority    = t["priority"]
        ttype       = t["type"]                   # "type" not "ticket_type"
        title_str   = t["title"]                  # "title" not "summary"
        technician  = t.get("technician", "Unassigned")
        hours       = t.get("hours_billed", 0.0)  # "hours_billed" not "hours_logged"
        bill_status = t.get("billing_status", "")
        sla_ok      = t.get("sla_met", True)
        created_dt  = t.get("created_date", "")
        resolved_dt = t.get("resolved_date")       # "resolved_date" not "closed_date"
        linked_inv  = t.get("linked_invoice", "")
        resolution  = t.get("resolution_note", "")

        sla_label = "MET" if sla_ok else "BREACHED"

        content = (
            f"SERVICE TICKET - TechVault MSP ConnectWise PSA\n"
            f"{'=' * 55}\n\n"
            f"Ticket ID       : {tid}\n"
            f"Client          : {client_name}\n"
            f"Type            : {ttype}\n"
            f"Priority        : {priority}\n"
            f"Status          : {status}\n"
            f"Assigned To     : {technician}\n\n"
            f"Title           : {title_str}\n\n"
            f"Created         : {created_dt}\n"
            f"Resolved        : {resolved_dt if resolved_dt else 'Open'}\n"
            f"Linked Invoice  : {linked_inv if linked_inv else 'None'}\n\n"
            f"SLA Compliance  : {sla_label}\n"
            f"Hours Billed    : {hours:.1f}h\n"
            f"Billing Status  : {bill_status}\n"
        )

        if resolution:
            content += f"\nResolution Notes: {resolution}\n"

        content += f"\nSource System   : {t.get('source_system', 'ConnectWise Manage')}\n"

        doc_title = f"Ticket {tid} - {client_name} - {priority} {ttype} - {status}"

        published_at: datetime | None = None
        try:
            published_at = datetime.fromisoformat(f"{created_dt}T00:00:00") if created_dt else None
        except Exception:
            pass

        return RawDocument(
            id=tid.lower().replace("-", "_"),
            source=f"psa:connectwise:{tid}",
            source_type=SourceType.PSA,
            title=doc_title,
            content=content,
            url=None,
            authors=[technician] if technician and technician != "Unassigned" else [],
            published_at=published_at,
            metadata={
                "ticket_id":      tid,
                "client_id":      t.get("client_id", ""),
                "client_name":    client_name,
                "ticket_type":    ttype,
                "priority":       priority,
                "status":         status,
                "sla_met":        sla_ok,
                "hours_billed":   hours,
                "billing_status": bill_status,
                "technician":     technician,
                "created_date":   created_dt,
                "resolved_date":  resolved_dt,
                "linked_invoice": linked_inv,
                "source_system":  "ConnectWise Manage",
                "source_system_type": "PSA",
            },
        )
