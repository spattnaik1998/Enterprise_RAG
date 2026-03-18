"""
Proactive Alert Generators
--------------------------
Three scheduled job functions that detect risks and return alert payloads.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

from src.observability.quality_monitor import QualityMonitor


def check_ar_risk() -> list[dict]:
    """
    Check for overdue invoices (>30 days) and cross-reference CRM health.

    Returns:
        [{alert_type: "ar_risk", severity: "high"|"medium", payload: {...}}]
    """
    alerts = []
    try:
        # Load invoices
        invoices_file = Path("data/enterprise/invoices.json")
        if not invoices_file.exists():
            logger.warning("[AR Risk] invoices.json not found")
            return alerts

        with open(invoices_file) as f:
            invoice_data = json.load(f)

        # Load CRM profiles for health lookup
        crm_file = Path("data/enterprise/crm_profiles.json")
        crm_health = {}
        if crm_file.exists():
            with open(crm_file) as f:
                crm_data = json.load(f)
                for profile in crm_data.get("records", []):
                    client_id = profile.get("client_id")
                    health = profile.get("account_health", "YELLOW")
                    crm_health[client_id] = health

        # Check each invoice
        today = datetime.now().date()
        for invoice in invoice_data.get("records", []):
            invoice_date_str = invoice.get("invoice_date", "")
            try:
                invoice_date = datetime.strptime(invoice_date_str, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue

            days_overdue = (today - invoice_date).days
            if days_overdue > 30:
                client_id = invoice.get("client_id", "unknown")
                amount = invoice.get("amount", 0)
                health = crm_health.get(client_id, "YELLOW")

                # Severity: RED health + overdue = high, otherwise medium
                severity = "high" if health == "RED" and days_overdue > 60 else "medium"

                alerts.append({
                    "alert_type": "ar_risk",
                    "severity": severity,
                    "payload": {
                        "client_id": client_id,
                        "amount": amount,
                        "days_overdue": days_overdue,
                        "health": health,
                        "invoice_id": invoice.get("id"),
                    }
                })

        logger.info(f"[AR Risk Check] Found {len(alerts)} overdue invoices")

    except Exception as e:
        logger.error(f"[AR Risk Check] Error: {e}")

    return alerts


def check_contract_expiry() -> list[dict]:
    """
    Check for contracts expiring in <90 days.

    Returns:
        [{alert_type: "contract_expiry", severity: "high"|"medium", payload: {...}}]
    """
    alerts = []
    try:
        contracts_file = Path("data/enterprise/contracts.json")
        if not contracts_file.exists():
            logger.warning("[Contract Expiry] contracts.json not found")
            return alerts

        with open(contracts_file) as f:
            contract_data = json.load(f)

        # Load CRM for health lookup
        crm_file = Path("data/enterprise/crm_profiles.json")
        crm_health = {}
        if crm_file.exists():
            with open(crm_file) as f:
                crm_data = json.load(f)
                for profile in crm_data.get("records", []):
                    client_id = profile.get("client_id")
                    health = profile.get("account_health", "YELLOW")
                    crm_health[client_id] = health

        today = datetime.now().date()
        for contract in contract_data.get("records", []):
            expiry_str = contract.get("expiry_date", "")
            try:
                expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue

            days_until_expiry = (expiry_date - today).days
            if 0 < days_until_expiry < 90:
                client_id = contract.get("client_id", "unknown")
                health = crm_health.get(client_id, "YELLOW")
                annual_value = contract.get("annual_value", 0)

                # Severity: RED health + imminent expiry = high
                severity = "high" if health == "RED" and days_until_expiry < 30 else "medium"

                alerts.append({
                    "alert_type": "contract_expiry",
                    "severity": severity,
                    "payload": {
                        "client_id": client_id,
                        "contract_id": contract.get("id"),
                        "days_until_expiry": days_until_expiry,
                        "annual_value": annual_value,
                        "health": health,
                        "expiry_date": expiry_str,
                    }
                })

        logger.info(f"[Contract Expiry Check] Found {len(alerts)} expiring contracts")

    except Exception as e:
        logger.error(f"[Contract Expiry Check] Error: {e}")

    return alerts


def check_quality_degradation() -> list[dict]:
    """
    Check QualityMonitor for degradation signals.

    Returns:
        [{alert_type: "quality_degradation", severity: "high"|"medium", payload: {...}}]
    """
    alerts = []
    try:
        monitor = QualityMonitor()
        degradation_signals = monitor.get_degradation_signal()

        for signal in degradation_signals:
            severity = "high" if signal.current_value < signal.threshold * 0.8 else "medium"
            alerts.append({
                "alert_type": "quality_degradation",
                "severity": severity,
                "payload": {
                    "metric": signal.alert_type,
                    "current_value": round(signal.current_value, 4),
                    "threshold": round(signal.threshold, 4),
                    "window_size": signal.window_size,
                }
            })

        logger.info(f"[Quality Degradation Check] Found {len(alerts)} quality issues")

    except Exception as e:
        logger.error(f"[Quality Degradation Check] Error: {e}")

    return alerts
