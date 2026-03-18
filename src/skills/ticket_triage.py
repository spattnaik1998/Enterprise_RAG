"""
Ticket Triage Skill
-------------------
Classify PSA tickets by SLA risk and recommend assignment.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.skills.base import Skill, SkillContext, SkillResult


class TicketTriageSkill(Skill):
    """Triage PSA tickets by SLA risk and client health."""

    name = "ticket_triage"
    description = "Triage PSA tickets by SLA urgency and client health, recommend assignment"
    required_sources = ["psa", "crm"]
    version = "1.0.0"

    async def execute(self, context: SkillContext) -> SkillResult:
        """Triage tickets."""
        start = time.time()

        try:
            # Load tickets
            psa_file = Path("data/enterprise/psa_tickets.json")
            if not psa_file.exists():
                return SkillResult(
                    success=False,
                    error="psa_tickets.json not found",
                    skill_name=self.name,
                )

            with open(psa_file) as f:
                psa_data = json.load(f)

            # Load CRM
            crm_file = Path("data/enterprise/crm_profiles.json")
            crm_health = {}
            if crm_file.exists():
                with open(crm_file) as f:
                    crm_data = json.load(f)
                    for profile in crm_data.get("records", []):
                        client_id = profile.get("client_id")
                        health = profile.get("account_health", "YELLOW")
                        crm_health[client_id] = health

            # Classify tickets
            tickets_by_priority = {"critical": [], "high": [], "medium": [], "low": []}

            for ticket in psa_data.get("records", []):
                ticket_type = ticket.get("type", "maintenance")
                ticket_status = ticket.get("status", "new")
                client_id = ticket.get("client_id", "unknown")
                health = crm_health.get(client_id, "YELLOW")

                # SLA risk: emergency type + RED health = critical
                priority = "low"
                if ticket_type in ["emergency", "outage"]:
                    if health == "RED":
                        priority = "critical"
                    elif health == "YELLOW":
                        priority = "high"
                    else:
                        priority = "medium"
                elif ticket_type == "incident" and health == "RED":
                    priority = "high"
                elif health == "RED":
                    priority = "medium"

                tickets_by_priority[priority].append({
                    "id": ticket.get("id"),
                    "client_id": client_id,
                    "title": ticket.get("title"),
                    "type": ticket_type,
                    "status": ticket_status,
                    "health": health,
                })

            # Generate triage report
            report_lines = [
                "# Ticket Triage Report",
                "",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## Summary",
                "",
            ]

            total = sum(len(v) for v in tickets_by_priority.values())
            critical_count = len(tickets_by_priority["critical"])
            high_count = len(tickets_by_priority["high"])

            report_lines.append(f"- **Total Open Tickets**: {total}")
            report_lines.append(f"- **Critical Priority**: {critical_count}")
            report_lines.append(f"- **High Priority**: {high_count}")
            report_lines.append("")

            for priority in ["critical", "high", "medium", "low"]:
                tickets = tickets_by_priority[priority]
                if tickets:
                    report_lines.append(f"## {priority.upper()} ({len(tickets)})")
                    report_lines.append("")
                    for ticket in tickets:
                        report_lines.append(f"- **{ticket['id']}** - {ticket['title']}")
                        report_lines.append(f"  - Client: {ticket['client_id']} ({ticket['health']})")
                        report_lines.append(f"  - Type: {ticket['type']} | Status: {ticket['status']}")
                    report_lines.append("")

            report = "\n".join(report_lines)

            latency_ms = (time.time() - start) * 1000

            return SkillResult(
                success=True,
                data={"report": report, "by_priority": tickets_by_priority},
                latency_ms=latency_ms,
                skill_name=self.name,
            )

        except Exception as e:
            logger.error(f"[TicketTriage] Error: {e}")
            latency_ms = (time.time() - start) * 1000
            return SkillResult(
                success=False,
                error=str(e),
                latency_ms=latency_ms,
                skill_name=self.name,
            )
