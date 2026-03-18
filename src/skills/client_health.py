"""
Client Health Skill
------------------
Compute composite client health score from billing, tickets, and CRM.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.skills.base import Skill, SkillContext, SkillResult


class ClientHealthSkill(Skill):
    """Compute composite client health score."""

    name = "client_health"
    description = "Compute composite client health score from billing, tickets, and relationships"
    required_sources = ["billing", "psa", "crm"]
    version = "1.0.0"

    async def execute(self, context: SkillContext) -> SkillResult:
        """Generate client health scores."""
        start = time.time()

        try:
            # Load data
            crm_file = Path("data/enterprise/crm_profiles.json")
            if not crm_file.exists():
                return SkillResult(
                    success=False,
                    error="crm_profiles.json not found",
                    skill_name=self.name,
                )

            with open(crm_file) as f:
                crm_data = json.load(f)

            # Load invoices for payment health
            invoices = {}
            invoices_file = Path("data/enterprise/invoices.json")
            if invoices_file.exists():
                with open(invoices_file) as f:
                    invoice_data = json.load(f)
                    for inv in invoice_data.get("records", []):
                        client_id = inv.get("client_id", "unknown")
                        if client_id not in invoices:
                            invoices[client_id] = {"total": 0, "overdue": 0}
                        invoices[client_id]["total"] += float(inv.get("amount", 0))

            # Load tickets for support health
            tickets = {}
            psa_file = Path("data/enterprise/psa_tickets.json")
            if psa_file.exists():
                with open(psa_file) as f:
                    psa_data = json.load(f)
                    for ticket in psa_data.get("records", []):
                        client_id = ticket.get("client_id", "unknown")
                        if client_id not in tickets:
                            tickets[client_id] = 0
                        tickets[client_id] += 1

            # Compute health scores
            today = datetime.now().date()
            report_lines = [
                "# Client Health Scorecard",
                "",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## Health by Client",
                "",
            ]

            results = []

            for profile in crm_data.get("records", []):
                client_id = profile.get("client_id", "unknown")
                crm_health = profile.get("account_health", "YELLOW")

                # Summarize data points
                annual_revenue = invoices.get(client_id, {}).get("total", 0)
                ticket_count = tickets.get(client_id, 0)

                results.append({
                    "client_id": client_id,
                    "crm_health": crm_health,
                    "annual_revenue": annual_revenue,
                    "ticket_count": ticket_count,
                    "contact_count": len([c for c in [profile.get("contacts", {}).get(role) for role in ["cfo", "it_manager", "ar_contact"]] if c]),
                })

            # Sort by health (RED first)
            results.sort(key=lambda x: (x["crm_health"] != "RED", -x["annual_revenue"]))

            for result in results:
                health_badge = {
                    "RED": "[RED]",
                    "YELLOW": "[YELLOW]",
                    "GREEN": "[GREEN]",
                }.get(result["crm_health"], "[?]")

                report_lines.append(f"### {result['client_id']} {health_badge}")
                report_lines.append(f"- **Annual Revenue**: ${result['annual_revenue']:,.2f}")
                report_lines.append(f"- **Open Tickets**: {result['ticket_count']}")
                report_lines.append(f"- **Key Contacts**: {result['contact_count']}")
                report_lines.append("")

            report = "\n".join(report_lines)

            latency_ms = (time.time() - start) * 1000

            return SkillResult(
                success=True,
                data={"report": report, "scores": results},
                latency_ms=latency_ms,
                skill_name=self.name,
            )

        except Exception as e:
            logger.error(f"[ClientHealth] Error: {e}")
            latency_ms = (time.time() - start) * 1000
            return SkillResult(
                success=False,
                error=str(e),
                latency_ms=latency_ms,
                skill_name=self.name,
            )
