"""
Communications Collector (Microsoft Exchange Online simulation)
----------------------------------------------------------------
Reads communications.json from data/enterprise/ and converts each
email/reminder record into a rich-text RawDocument for RAG indexing.

Each document captures:
  - Message header (ID, direction, channel, dates)
  - Invoice context (which invoice, amount, days overdue at send time)
  - Reminder sequence position (1st notice -> 4th escalation)
  - Message subject and body preview
  - Client response classification
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from loguru import logger

from src.collection.base_collector import BaseCollector
from src.schemas import RawDocument, SourceType


class CommsCollector(BaseCollector):
    """
    Yields one RawDocument per email record from the Exchange Online export.

    The communications log is a critical data source for understanding the
    collection history of overdue invoices: has a reminder been sent?
    Has the client responded? How many notices have been issued?
    """

    source_type = SourceType.COMMUNICATIONS

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.data_file = Path(config.get("data_file", "data/enterprise/communications.json"))

    async def health_check(self) -> bool:
        ok = self.data_file.exists()
        if not ok:
            logger.warning(f"[Comms] Data file not found: {self.data_file}")
        return ok

    async def collect(self) -> AsyncIterator[RawDocument]:
        logger.info(f"[Comms] Loading communications from {self.data_file}")
        try:
            raw = json.loads(self.data_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error(f"[Comms] Failed to read data file: {exc}")
            return

        records = raw if isinstance(raw, list) else raw.get("records", [])
        logger.info(f"[Comms] Processing {len(records)} communication records")

        for record in records:
            try:
                yield self._to_document(record)
            except Exception as exc:
                logger.warning(f"[Comms] Skipping record {record.get('comm_id', '?')}: {exc}")

    # --- Document construction ------------------------------------------------

    def _to_document(self, c: dict) -> RawDocument:
        comm_id    = c["comm_id"]
        inv_id     = c["invoice_id"]
        client_name = c["client_name"]
        sent_date  = c["sent_date"]
        seq        = c["reminder_sequence"]
        direction  = c.get("direction", "OUTBOUND")
        sent_from  = c.get("sent_from", "")
        sent_to    = c.get("sent_to", "")
        subject    = c.get("subject", "")
        preview    = c.get("body_preview", "")
        status     = c.get("status", "DELIVERED")
        response   = c.get("client_response", "NO_RESPONSE")
        amount     = c.get("invoice_amount", 0.0)
        balance    = c.get("balance_at_time", 0.0)
        days_over  = c.get("days_overdue_at_send", 0)
        am         = c.get("account_manager", "")

        # Sequence label
        seq_labels = {
            1: "1st Notice (Friendly Reminder - 10 days past due)",
            2: "2nd Notice (30 days past due)",
            3: "3rd Notice / Final Warning (60 days past due)",
            4: "4th Notice / Escalation (90+ days - Collections Review)",
        }
        seq_label = seq_labels.get(seq, f"Notice #{seq}")

        # Response prose
        response_prose = {
            "PAYMENT_PROMISE":  "Client promised payment by a specific date",
            "PARTIAL_PAYMENT":  "Client made a partial payment",
            "PAYMENT_RECEIVED": "Payment was received in full",
            "DISPUTE":          "Client disputed the invoice",
            "NO_RESPONSE":      "No response received from client",
            "OUT_OF_OFFICE":    "Client auto-replied as out of office",
        }.get(response, response)

        content = (
            f"COMMUNICATIONS RECORD - TechVault MSP Exchange Online\n"
            f"{'=' * 55}\n\n"
            f"Comm ID         : {comm_id}\n"
            f"Invoice Ref     : {inv_id}\n"
            f"Client          : {client_name}\n"
            f"Account Manager : {am}\n\n"
            f"Notice Sequence : {seq_label}\n"
            f"Direction       : {direction}\n"
            f"Sent Date       : {sent_date[:10]}\n"
            f"Status          : {status}\n\n"
            f"FROM            : {sent_from}\n"
            f"TO              : {sent_to}\n"
            f"SUBJECT         : {subject}\n\n"
            f"Invoice Amount  : ${amount:,.2f}\n"
            f"Balance at Send : ${balance:,.2f}\n"
            f"Days Overdue    : {days_over}\n\n"
            f"MESSAGE PREVIEW:\n"
            f"----------------\n"
            f"{preview}\n\n"
            f"CLIENT RESPONSE : {response_prose}\n\n"
            f"Source System   : {c.get('source_system', 'Microsoft Exchange Online')}\n"
        )

        title = (
            f"Comms {comm_id} - {seq_label[:20]} - {client_name} re {inv_id}"
        )

        published_at: datetime | None = None
        try:
            published_at = datetime.fromisoformat(sent_date)
        except Exception:
            pass

        return RawDocument(
            id=f"comm_{comm_id.lower().replace('-', '_')}",
            source=f"communications:exchange:{comm_id}",
            source_type=SourceType.COMMUNICATIONS,
            title=title,
            content=content,
            url=None,
            authors=[am] if am else [],
            published_at=published_at,
            metadata={
                "comm_id":           comm_id,
                "invoice_id":        inv_id,
                "client_id":         c.get("client_id", ""),
                "client_name":       client_name,
                "reminder_sequence": seq,
                "direction":         direction,
                "sent_date":         sent_date,
                "client_response":   response,
                "days_overdue_at_send": days_over,
                "invoice_amount":    amount,
                "balance_at_time":   balance,
                "account_manager":   am,
                "source_system":     "Microsoft Exchange Online",
                "source_system_type": "Communications",
            },
        )
