"""
CRM Collector (HubSpot CRM simulation)
----------------------------------------
Reads crm_profiles.json from data/enterprise/ and converts each client
profile into a rich-text RawDocument for RAG indexing.

Each document captures:
  - Client identity (name, industry, employee count)
  - Account health (RED/YELLOW/GREEN, health note)
  - Financial relationship (open balance, YTD billed/paid)
  - Upsell opportunities flagged by account managers
  - Key contacts (CFO, IT manager, AR contact)
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from loguru import logger

from src.collection.base_collector import BaseCollector
from src.schemas import RawDocument, SourceType


class CRMCollector(BaseCollector):
    """
    Yields one RawDocument per client profile from the HubSpot-style CRM export.
    """

    source_type = SourceType.CRM

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.data_file = Path(config.get("data_file", "data/enterprise/crm_profiles.json"))

    async def health_check(self) -> bool:
        ok = self.data_file.exists()
        if not ok:
            logger.warning(f"[CRM] Data file not found: {self.data_file}")
        return ok

    async def collect(self) -> AsyncIterator[RawDocument]:
        logger.info(f"[CRM] Loading profiles from {self.data_file}")
        try:
            raw = json.loads(self.data_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error(f"[CRM] Failed to read data file: {exc}")
            return

        profiles = raw if isinstance(raw, list) else raw.get("records", [])
        logger.info(f"[CRM] Processing {len(profiles)} client profiles")

        for profile in profiles:
            try:
                yield self._to_document(profile)
            except Exception as exc:
                logger.warning(f"[CRM] Skipping profile {profile.get('client_id', '?')}: {exc}")

    # --- Document construction ------------------------------------------------

    def _to_document(self, p: dict) -> RawDocument:
        cid         = p["client_id"]
        name        = p["client_name"]
        industry    = p.get("industry", "")
        emp_count   = p.get("employee_count", 0)
        since       = p.get("client_since", "")
        years       = p.get("years_as_client", 0)
        am          = p.get("account_manager", "")
        health      = p.get("account_health", "UNKNOWN")  # RED/YELLOW/GREEN
        health_note = p.get("health_note", "")
        reliability = p.get("payment_reliability", "UNKNOWN")
        billed_ytd  = p.get("total_billed_ytd", 0.0)
        paid_ytd    = p.get("total_paid_ytd", 0.0)
        open_bal    = p.get("open_balance", 0.0)
        overdue_cnt = p.get("overdue_invoice_count", 0)
        overdue_bal = p.get("overdue_balance", 0.0)
        upsell      = p.get("upsell_opportunities", [])
        last_qbr    = p.get("last_qbr_date", "N/A")
        nps         = p.get("nps_score", 0)
        contacts    = p.get("contacts", {})
        address     = p.get("address", {})

        # NPS sentiment
        if nps >= 9:
            nps_label = "Promoter"
        elif nps >= 7:
            nps_label = "Passive"
        else:
            nps_label = "Detractor"

        # Upsell list
        upsell_text = "\n  - ".join(upsell) if upsell else "None identified"

        # Contacts
        cfo     = contacts.get("cfo", {})
        it_mgr  = contacts.get("it_manager", {})
        ar_cont = contacts.get("ar_contact", {})

        content = (
            f"CLIENT PROFILE - TechVault MSP HubSpot CRM\n"
            f"{'=' * 55}\n\n"
            f"Client ID       : {cid}\n"
            f"Company         : {name}\n"
            f"Industry        : {industry}\n"
            f"Employees       : {emp_count}\n"
            f"Client Since    : {since} ({years:.1f} years)\n"
            f"Location        : {address.get('city', '')}, {address.get('state', '')}\n\n"
            f"Account Manager : {am}\n"
            f"Last QBR        : {last_qbr}\n\n"
            f"ACCOUNT HEALTH\n"
            f"--------------\n"
            f"Health Status   : {health}\n"
            f"Health Note     : {health_note}\n"
            f"Payment Reliability: {reliability}\n"
            f"NPS Score       : {nps}/10 ({nps_label})\n\n"
            f"FINANCIAL SUMMARY\n"
            f"-----------------\n"
            f"YTD Billed      : ${billed_ytd:,.2f}\n"
            f"YTD Paid        : ${paid_ytd:,.2f}\n"
            f"Open Balance    : ${open_bal:,.2f}\n"
            f"Overdue Invoices: {overdue_cnt} invoice(s) totaling ${overdue_bal:,.2f}\n\n"
            f"UPSELL OPPORTUNITIES\n"
            f"--------------------\n"
            f"  - {upsell_text}\n\n"
            f"KEY CONTACTS\n"
            f"------------\n"
            f"CFO             : {cfo.get('name', 'N/A')} - {cfo.get('email', '')}\n"
            f"IT Manager      : {it_mgr.get('name', 'N/A')} - {it_mgr.get('email', '')}\n"
            f"AR Contact      : {ar_cont.get('name', 'N/A')} - {ar_cont.get('email', '')}\n\n"
            f"Source System   : {p.get('source_system', 'HubSpot CRM')}\n"
        )

        title = (
            f"CRM Profile: {name} - {industry} - "
            f"Health {health} - NPS {nps}/10"
        )

        published_at: datetime | None = None
        try:
            published_at = datetime.fromisoformat(f"{since}T00:00:00") if since else None
        except Exception:
            pass

        return RawDocument(
            id=f"crm_{cid.lower()}",
            source=f"crm:hubspot:{cid}",
            source_type=SourceType.CRM,
            title=title,
            content=content,
            url=None,
            authors=[am] if am else [],
            published_at=published_at,
            metadata={
                "client_id":         cid,
                "client_name":       name,
                "industry":          industry,
                "employee_count":    emp_count,
                "account_health":    health,
                "payment_reliability": reliability,
                "open_balance":      open_bal,
                "overdue_balance":   overdue_bal,
                "overdue_count":     overdue_cnt,
                "nps_score":         nps,
                "account_manager":   am,
                "upsell_count":      len(upsell),
                "source_system":     "HubSpot CRM",
                "source_system_type": "CRM",
            },
        )
