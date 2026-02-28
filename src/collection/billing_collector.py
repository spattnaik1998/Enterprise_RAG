"""
Billing Collector (QuickBooks Enterprise simulation)
-----------------------------------------------------
Reads invoices.json from data/enterprise/ and converts each invoice
record into a rich-text RawDocument suitable for RAG indexing.

Each document captures:
  - Invoice header (ID, dates, status, amounts)
  - Client info (name, account manager)
  - Line-item services rendered
  - Payment history and aging analysis
  - Late fee accrual
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from loguru import logger

from src.collection.base_collector import BaseCollector
from src.schemas import RawDocument, SourceType


class BillingCollector(BaseCollector):
    """
    Yields one RawDocument per invoice from the QuickBooks-style billing export.

    The content field is a natural-language narrative so that semantic search
    ("which invoices are 90 days overdue?") retrieves the right documents.
    """

    source_type = SourceType.BILLING

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.data_file = Path(config.get("data_file", "data/enterprise/invoices.json"))

    async def health_check(self) -> bool:
        ok = self.data_file.exists()
        if not ok:
            logger.warning(f"[Billing] Data file not found: {self.data_file}")
        return ok

    async def collect(self) -> AsyncIterator[RawDocument]:
        logger.info(f"[Billing] Loading invoices from {self.data_file}")
        try:
            raw = json.loads(self.data_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error(f"[Billing] Failed to read data file: {exc}")
            return

        invoices = raw if isinstance(raw, list) else raw.get("records", [])
        logger.info(f"[Billing] Processing {len(invoices)} invoice records")

        for inv in invoices:
            try:
                yield self._to_document(inv)
            except Exception as exc:
                logger.warning(f"[Billing] Skipping invoice {inv.get('invoice_id', '?')}: {exc}")

    # --- Document construction ------------------------------------------------

    def _to_document(self, inv: dict) -> RawDocument:
        inv_id      = inv["invoice_id"]
        client_name = inv["client_name"]
        status      = inv["status"]
        total       = inv["total_amount"]
        balance     = inv["balance_due"]
        days_out    = inv["days_outstanding"]
        issue_dt    = inv["invoice_date"]      # field is "invoice_date" in the data
        due_dt      = inv["due_date"]
        period      = inv.get("billing_period", "")
        inv_type    = inv.get("invoice_type", "")
        am          = inv.get("account_manager", "")
        terms       = inv.get("payment_terms", "NET30")
        late_fee    = inv.get("late_fee_amount", 0.0)

        # Status prose for semantic retrieval
        status_prose = {
            "PAID":             "has been paid in full",
            "OPEN":             "is open and within payment terms",
            "PAST_DUE":         f"is past due by {days_out} days",
            "OVERDUE":          f"is OVERDUE by {days_out} days and requires collection action",
            "OVERDUE_CRITICAL": f"is CRITICALLY OVERDUE by {days_out} days and has been escalated",
        }.get(status, f"has status {status}")

        # Build line-items section
        lines_text = ""
        for item in inv.get("line_items", []):
            lines_text += (
                f"\n  - {item['description']}: "
                f"{item['quantity']} x ${item['unit_price']:,.2f} = ${item['amount']:,.2f}"
            )

        # Aging bucket label
        if days_out == 0:
            aging = "Current"
        elif days_out <= 30:
            aging = "1-30 days"
        elif days_out <= 60:
            aging = "31-60 days"
        elif days_out <= 90:
            aging = "61-90 days"
        else:
            aging = "90+ days (Critical)"

        content = (
            f"INVOICE RECORD - TechVault MSP Billing System\n"
            f"{'=' * 55}\n\n"
            f"Invoice ID      : {inv_id}\n"
            f"Invoice Type    : {inv_type}\n"
            f"Client          : {client_name}\n"
            f"Account Manager : {am}\n\n"
            f"Invoice Date    : {issue_dt}\n"
            f"Due Date        : {due_dt}\n"
            f"Billing Period  : {period}\n"
            f"Payment Terms   : {terms}\n\n"
            f"Subtotal        : ${inv.get('subtotal', 0):,.2f}\n"
            f"Tax ({inv.get('tax_rate', 0) * 100:.0f}%)         : ${inv.get('tax_amount', 0):,.2f}\n"
            f"Invoice Total   : ${total:,.2f}\n"
            f"Amount Paid     : ${inv.get('amount_paid', 0):,.2f}\n"
            f"Balance Due     : ${balance:,.2f}\n"
            f"Days Outstanding: {days_out}\n"
            f"Aging Bucket    : {aging}\n\n"
            f"STATUS: This invoice {status_prose}.\n\n"
            f"Services Rendered:{lines_text if lines_text else ' (no line items)'}\n\n"
            f"Billing Contact : {inv.get('billing_contact', 'N/A')}\n"
            f"Source System   : {inv.get('source_system', 'QuickBooks Enterprise')}\n"
        )

        # Add late fee notice if applicable
        if late_fee and late_fee > 0:
            content += (
                f"\nLate Fee Accrued: ${late_fee:,.2f} "
                f"(1.5% per month on outstanding balance)\n"
            )

        if inv.get("notes"):
            content += f"\nNotes: {inv['notes']}\n"

        title = (
            f"Invoice {inv_id} - {client_name} - "
            f"${total:,.2f} - {status} ({days_out}d outstanding)"
        )

        published_at: datetime | None = None
        try:
            published_at = datetime.fromisoformat(f"{issue_dt}T00:00:00")
        except Exception:
            pass

        return RawDocument(
            id=inv_id.lower().replace("-", "_") + "_" + inv.get("client_id", "")[:6].lower(),
            source=f"billing:quickbooks:{inv_id}",
            source_type=SourceType.BILLING,
            title=title,
            content=content,
            url=None,
            authors=[am] if am else [],
            published_at=published_at,
            metadata={
                "invoice_id":       inv_id,
                "client_id":        inv.get("client_id", ""),
                "client_name":      client_name,
                "invoice_type":     inv_type,
                "status":           status,
                "total_amount":     total,
                "balance_due":      balance,
                "days_outstanding": days_out,
                "aging_bucket":     aging,
                "invoice_date":     issue_dt,
                "due_date":         due_dt,
                "billing_period":   period,
                "account_manager":  am,
                "late_fee_amount":  late_fee,
                "source_system":    "QuickBooks Enterprise",
                "source_system_type": "Billing",
            },
        )
