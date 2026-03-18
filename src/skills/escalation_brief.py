"""
Escalation Brief Skill
---------------------
Compile client 360 data into escalation document.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.skills.base import Skill, SkillContext, SkillResult


class EscalationBriefSkill(Skill):
    """Compile client 360 data into escalation document."""

    name = "escalation_brief"
    description = "Compile client 360 data and relationship context for escalations"
    required_sources = ["crm", "billing", "psa", "contracts"]
    version = "1.0.0"

    async def execute(self, context: SkillContext) -> SkillResult:
        """Generate escalation brief."""
        start = time.time()

        try:
            # Extract client ID from context
            query_lower = context.query.lower()
            client_id = None
            for word in context.query.split():
                if word.replace("?", "").isalnum():
                    client_id = word.replace("?", "")
                    break

            if not client_id:
                return SkillResult(
                    success=False,
                    error="Client ID not found in query",
                    skill_name=self.name,
                )

            # Load all data sources
            crm_profile = None
            invoices = []
            tickets = []
            contracts = []

            # CRM
            crm_file = Path("data/enterprise/crm_profiles.json")
            if crm_file.exists():
                with open(crm_file) as f:
                    crm_data = json.load(f)
                    for profile in crm_data.get("records", []):
                        if profile.get("client_id") == client_id:
                            crm_profile = profile
                            break

            # Invoices
            invoices_file = Path("data/enterprise/invoices.json")
            if invoices_file.exists():
                with open(invoices_file) as f:
                    invoice_data = json.load(f)
                    invoices = [inv for inv in invoice_data.get("records", [])
                                if inv.get("client_id") == client_id]

            # Tickets
            psa_file = Path("data/enterprise/psa_tickets.json")
            if psa_file.exists():
                with open(psa_file) as f:
                    psa_data = json.load(f)
                    tickets = [t for t in psa_data.get("records", [])
                               if t.get("client_id") == client_id]

            # Contracts
            contracts_file = Path("data/enterprise/contracts.json")
            if contracts_file.exists():
                with open(contracts_file) as f:
                    contract_data = json.load(f)
                    contracts = [c for c in contract_data.get("records", [])
                                 if c.get("client_id") == client_id]

            # Generate escalation brief
            report_lines = [
                f"# Escalation Brief: {client_id}",
                "",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## Client Summary",
                "",
            ]

            if crm_profile:
                health = crm_profile.get("account_health", "UNKNOWN")
                report_lines.append(f"- **Health Status**: {health}")
                report_lines.append(f"- **Industry**: {crm_profile.get('industry', 'N/A')}")
                report_lines.append(f"- **Employees**: {crm_profile.get('employee_count', 'N/A')}")

                contacts = crm_profile.get("contacts", {})
                if any(contacts.values()):
                    report_lines.append("- **Key Contacts**:")
                    for role, name in contacts.items():
                        if name:
                            report_lines.append(f"  - {role}: {name}")

            report_lines.append("")
            report_lines.append("## Relationship Summary")
            report_lines.append("")
            report_lines.append(f"- **Total Revenue (YTD)**: ${sum(float(inv.get('amount', 0)) for inv in invoices):,.2f}")
            report_lines.append(f"- **Open Tickets**: {len([t for t in tickets if t.get('status') != 'resolved'])}")
            report_lines.append(f"- **Active Contracts**: {len(contracts)}")
            report_lines.append("")

            if tickets:
                report_lines.append("## Recent Tickets")
                report_lines.append("")
                for ticket in tickets[:5]:
                    report_lines.append(f"- {ticket.get('title')} ({ticket.get('status')})")

            report = "\n".join(report_lines)

            latency_ms = (time.time() - start) * 1000

            return SkillResult(
                success=True,
                data={
                    "report": report,
                    "client_id": client_id,
                    "invoices": invoices,
                    "tickets": tickets,
                    "contracts": contracts,
                },
                latency_ms=latency_ms,
                skill_name=self.name,
            )

        except Exception as e:
            logger.error(f"[EscalationBrief] Error: {e}")
            latency_ms = (time.time() - start) * 1000
            return SkillResult(
                success=False,
                error=str(e),
                latency_ms=latency_ms,
                skill_name=self.name,
            )
