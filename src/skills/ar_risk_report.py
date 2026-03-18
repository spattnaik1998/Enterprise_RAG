"""
AR Risk Report Skill
-------------------
Generates AR aging analysis with client health cross-reference.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.skills.base import Skill, SkillContext, SkillResult


class ARRiskReportSkill(Skill):
    """Generate AR aging analysis with health scoring."""

    name = "ar_risk_report"
    description = "Generate AR aging analysis with client health cross-reference and risk scoring"
    required_sources = ["billing", "crm"]
    version = "1.0.0"

    async def execute(self, context: SkillContext) -> SkillResult:
        """Generate AR risk report."""
        start = time.time()

        try:
            # Load invoices
            invoices_file = Path("data/enterprise/invoices.json")
            if not invoices_file.exists():
                return SkillResult(
                    success=False,
                    error="invoices.json not found",
                    skill_name=self.name,
                )

            with open(invoices_file) as f:
                invoice_data = json.load(f)

            # Load CRM profiles
            crm_file = Path("data/enterprise/crm_profiles.json")
            crm_health = {}
            if crm_file.exists():
                with open(crm_file) as f:
                    crm_data = json.load(f)
                    for profile in crm_data.get("records", []):
                        client_id = profile.get("client_id")
                        health = profile.get("account_health", "YELLOW")
                        crm_health[client_id] = health

            # Compute AR aging by client
            today = datetime.now().date()
            ar_buckets = {}

            for invoice in invoice_data.get("records", []):
                try:
                    invoice_date = datetime.strptime(invoice.get("invoice_date", ""), "%Y-%m-%d").date()
                except (ValueError, TypeError):
                    continue

                days_overdue = (today - invoice_date).days
                client_id = invoice.get("client_id", "unknown")
                amount = float(invoice.get("amount", 0))

                if client_id not in ar_buckets:
                    ar_buckets[client_id] = {
                        "client_id": client_id,
                        "health": crm_health.get(client_id, "YELLOW"),
                        "current": 0,
                        "30_60": 0,
                        "60_90": 0,
                        "90_plus": 0,
                        "total": 0,
                        "count": 0,
                    }

                ar_buckets[client_id]["total"] += amount
                ar_buckets[client_id]["count"] += 1

                if days_overdue <= 30:
                    ar_buckets[client_id]["current"] += amount
                elif days_overdue <= 60:
                    ar_buckets[client_id]["30_60"] += amount
                elif days_overdue <= 90:
                    ar_buckets[client_id]["60_90"] += amount
                else:
                    ar_buckets[client_id]["90_plus"] += amount

            # Generate markdown report
            report_lines = [
                "# AR Risk Report",
                "",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Analysis Date: {today}",
                "",
                "## Executive Summary",
                "",
            ]

            total_ar = sum(b["total"] for b in ar_buckets.values())
            high_risk = sum(b["total"] for b in ar_buckets.values() if b["health"] == "RED")

            report_lines.append(f"- **Total Outstanding AR**: ${total_ar:,.2f}")
            report_lines.append(f"- **At-Risk AR (RED clients)**: ${high_risk:,.2f}")
            report_lines.append(f"- **90+ Days Overdue**: ${sum(b['90_plus'] for b in ar_buckets.values()):,.2f}")
            report_lines.append("")
            report_lines.append("## Aging Analysis by Client")
            report_lines.append("")

            # Sort by health risk (RED first), then by total AR
            sorted_clients = sorted(
                ar_buckets.values(),
                key=lambda x: (x["health"] != "RED", -x["total"]),
            )

            for bucket in sorted_clients:
                if bucket["total"] > 0:
                    health_badge = {
                        "RED": "[RED]",
                        "YELLOW": "[YELLOW]",
                        "GREEN": "[GREEN]",
                    }.get(bucket["health"], "[?]")

                    report_lines.append(f"### {bucket['client_id']} {health_badge}")
                    report_lines.append(f"- **Total**: ${bucket['total']:,.2f}")
                    report_lines.append(f"- **Current (0-30)**: ${bucket['current']:,.2f}")
                    report_lines.append(f"- **30-60 Days**: ${bucket['30_60']:,.2f}")
                    report_lines.append(f"- **60-90 Days**: ${bucket['60_90']:,.2f}")
                    report_lines.append(f"- **90+ Days**: ${bucket['90_plus']:,.2f}")
                    report_lines.append("")

            report = "\n".join(report_lines)

            latency_ms = (time.time() - start) * 1000

            return SkillResult(
                success=True,
                data={"report": report, "ar_buckets": ar_buckets},
                latency_ms=latency_ms,
                skill_name=self.name,
            )

        except Exception as e:
            logger.error(f"[ARRiskReport] Error: {e}")
            latency_ms = (time.time() - start) * 1000
            return SkillResult(
                success=False,
                error=str(e),
                latency_ms=latency_ms,
                skill_name=self.name,
            )
