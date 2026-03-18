"""
Contract Renewal Skill
---------------------
Identify upcoming renewals and draft renewal briefs.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

from src.skills.base import Skill, SkillContext, SkillResult


class ContractRenewalSkill(Skill):
    """Identify contract renewals and draft renewal briefs."""

    name = "contract_renewal"
    description = "Identify contracts due for renewal and draft renewal briefs"
    required_sources = ["contracts", "crm", "billing"]
    version = "1.0.0"

    async def execute(self, context: SkillContext) -> SkillResult:
        """Generate contract renewal brief."""
        start = time.time()

        try:
            # Load contracts
            contracts_file = Path("data/enterprise/contracts.json")
            if not contracts_file.exists():
                return SkillResult(
                    success=False,
                    error="contracts.json not found",
                    skill_name=self.name,
                )

            with open(contracts_file) as f:
                contract_data = json.load(f)

            # Load CRM
            crm_health = {}
            crm_file = Path("data/enterprise/crm_profiles.json")
            if crm_file.exists():
                with open(crm_file) as f:
                    crm_data = json.load(f)
                    for profile in crm_data.get("records", []):
                        client_id = profile.get("client_id")
                        health = profile.get("account_health", "YELLOW")
                        crm_health[client_id] = health

            # Identify renewals
            today = datetime.now().date()
            upcoming = []

            for contract in contract_data.get("records", []):
                try:
                    expiry = datetime.strptime(contract.get("expiry_date", ""), "%Y-%m-%d").date()
                except (ValueError, TypeError):
                    continue

                days_until = (expiry - today).days
                if 0 <= days_until <= 180:
                    client_id = contract.get("client_id", "unknown")
                    upcoming.append({
                        "id": contract.get("id"),
                        "client_id": client_id,
                        "expiry_date": contract.get("expiry_date"),
                        "days_until": days_until,
                        "annual_value": contract.get("annual_value", 0),
                        "health": crm_health.get(client_id, "YELLOW"),
                        "sla": contract.get("sla_response_time", {}),
                        "auto_renew": contract.get("auto_renew", False),
                    })

            # Sort by urgency (fewest days first, RED clients first)
            upcoming.sort(key=lambda x: (x["health"] != "RED", x["days_until"]))

            # Generate brief
            report_lines = [
                "# Contract Renewal Brief",
                "",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"## Upcoming Renewals (Next 180 Days)",
                "",
            ]

            total_value = sum(c["annual_value"] for c in upcoming)
            report_lines.append(f"- **Total Annual Value at Risk**: ${total_value:,.2f}")
            report_lines.append(f"- **Contracts Due**: {len(upcoming)}")
            report_lines.append("")

            for contract in upcoming:
                risk_level = "CRITICAL" if contract["health"] == "RED" and contract["days_until"] < 30 else "STANDARD"
                report_lines.append(f"### {contract['client_id']} - {contract['id']}")
                report_lines.append(f"- **Expiry**: {contract['expiry_date']} ({contract['days_until']} days)")
                report_lines.append(f"- **Annual Value**: ${contract['annual_value']:,.2f}")
                report_lines.append(f"- **Client Health**: {contract['health']}")
                report_lines.append(f"- **Auto-Renew**: {contract['auto_renew']}")
                report_lines.append(f"- **Risk Level**: {risk_level}")
                report_lines.append("")

            report = "\n".join(report_lines)

            latency_ms = (time.time() - start) * 1000

            return SkillResult(
                success=True,
                data={"report": report, "renewals": upcoming},
                latency_ms=latency_ms,
                skill_name=self.name,
            )

        except Exception as e:
            logger.error(f"[ContractRenewal] Error: {e}")
            latency_ms = (time.time() - start) * 1000
            return SkillResult(
                success=False,
                error=str(e),
                latency_ms=latency_ms,
                skill_name=self.name,
            )
