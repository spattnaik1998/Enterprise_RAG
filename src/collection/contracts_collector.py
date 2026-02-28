"""
Contracts Collector (SharePoint Document Repository simulation)
----------------------------------------------------------------
Reads contracts.json from data/enterprise/ and converts each service
agreement into a rich-text RawDocument for RAG indexing.

Each document captures:
  - Agreement identity and term dates
  - SLA commitments (response times by priority, uptime guarantee)
  - Payment terms and late payment penalty clause
  - Services in scope
  - Termination and auto-renewal provisions
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from loguru import logger

from src.collection.base_collector import BaseCollector
from src.schemas import RawDocument, SourceType


class ContractsCollector(BaseCollector):
    """
    Yields one RawDocument per contract from the SharePoint document export.
    """

    source_type = SourceType.CONTRACTS

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.data_file = Path(config.get("data_file", "data/enterprise/contracts.json"))

    async def health_check(self) -> bool:
        ok = self.data_file.exists()
        if not ok:
            logger.warning(f"[Contracts] Data file not found: {self.data_file}")
        return ok

    async def collect(self) -> AsyncIterator[RawDocument]:
        logger.info(f"[Contracts] Loading contracts from {self.data_file}")
        try:
            raw = json.loads(self.data_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error(f"[Contracts] Failed to read data file: {exc}")
            return

        contracts = raw if isinstance(raw, list) else raw.get("records", [])
        logger.info(f"[Contracts] Processing {len(contracts)} contract records")

        for contract in contracts:
            try:
                yield self._to_document(contract)
            except Exception as exc:
                logger.warning(
                    f"[Contracts] Skipping contract {contract.get('contract_id', '?')}: {exc}"
                )

    # --- Document construction ------------------------------------------------

    def _to_document(self, c: dict) -> RawDocument:
        cid         = c["contract_id"]
        client_name = c["client_name"]
        ctype       = c.get("contract_type", "")
        start       = c.get("effective_date", "")   # "effective_date" in data
        end         = c.get("expiry_date", "")       # "expiry_date" in data
        auto_renew  = c.get("auto_renew", False)     # "auto_renew" in data
        notice_days = c.get("renewal_notice_days", 90)
        payment_terms = c.get("payment_terms", "NET30")
        mrr         = c.get("monthly_value", 0.0)    # "monthly_value" in data
        arr         = c.get("annual_value", 0.0)     # "annual_value" in data
        services    = c.get("services_in_scope", []) # "services_in_scope" in data
        sla_resp    = c.get("sla_response_time", {}) # dict of priority -> response time
        sla_uptime  = c.get("sla_uptime_guarantee", "")
        late_penalty = c.get("late_payment_penalty", "")
        suspension  = c.get("suspension_clause", "")
        termination = c.get("termination_notice", "")
        governing   = c.get("governing_law", "")
        signed_client = c.get("signed_by_client", "")
        signed_msp  = c.get("signed_by_msp", "")

        # SLA response times table
        sla_text = ""
        if isinstance(sla_resp, dict):
            for priority, resp_time in sla_resp.items():
                sla_text += f"\n  - {priority}: {resp_time} response"
        else:
            sla_text = f"\n  {sla_resp}"

        # Services list
        services_text = "\n  - ".join(services) if services else "See contract schedule"

        content = (
            f"SERVICE AGREEMENT - TechVault MSP SharePoint Repository\n"
            f"{'=' * 55}\n\n"
            f"Contract ID     : {cid}\n"
            f"Client          : {client_name}\n"
            f"Contract Type   : {ctype}\n\n"
            f"Effective Date  : {start}\n"
            f"Expiry Date     : {end}\n"
            f"Auto-Renewal    : {'Yes - renews ' + str(notice_days) + ' days before expiry' if auto_renew else 'No - requires manual renewal'}\n\n"
            f"FINANCIAL TERMS\n"
            f"---------------\n"
            f"Monthly Value (MRR): ${mrr:,.2f}\n"
            f"Annual Value (ACV) : ${arr:,.2f}\n"
            f"Payment Terms      : {payment_terms}\n"
            f"Late Payment Penalty: {late_penalty}\n\n"
            f"SLA COMMITMENTS\n"
            f"---------------"
            f"{sla_text}\n"
            f"Uptime Guarantee   : {sla_uptime}\n\n"
            f"SERVICES IN SCOPE\n"
            f"-----------------\n"
            f"  - {services_text}\n\n"
            f"SUSPENSION CLAUSE\n"
            f"-----------------\n"
            f"{suspension}\n\n"
            f"TERMINATION TERMS\n"
            f"-----------------\n"
            f"{termination}\n\n"
            f"SIGNATORIES\n"
            f"-----------\n"
            f"Client Signatory: {signed_client}\n"
            f"MSP Signatory   : {signed_msp}\n"
            f"Governing Law   : {governing}\n\n"
            f"Source System   : {c.get('source_system', 'SharePoint Contract Repository')}\n"
        )

        title = (
            f"Contract {cid} - {client_name} - {ctype} - "
            f"${mrr:,.2f}/mo"
        )

        published_at: datetime | None = None
        try:
            published_at = datetime.fromisoformat(f"{start}T00:00:00") if start else None
        except Exception:
            pass

        return RawDocument(
            id=f"contract_{cid.lower().replace('-', '_')}",
            source=f"contracts:sharepoint:{cid}",
            source_type=SourceType.CONTRACTS,
            title=title,
            content=content,
            url=None,
            authors=[],
            published_at=published_at,
            metadata={
                "contract_id":    cid,
                "client_id":      c.get("client_id", ""),
                "client_name":    client_name,
                "contract_type":  ctype,
                "effective_date": start,
                "expiry_date":    end,
                "auto_renew":     auto_renew,
                "monthly_value":  mrr,
                "annual_value":   arr,
                "payment_terms":  payment_terms,
                "source_system":  "SharePoint Contract Repository",
                "source_system_type": "Contracts",
            },
        )
