"""
backfill_service_tickets.py
----------------------------
One-time backfill: copies all rows from the `psa_tickets` table (ConnectWise
historical data, populated by migrate_to_supabase.py) into the `service_tickets`
table (what the MSP admin dashboard reads via /api/msp/tickets).

Schema mapping:
  psa_tickets.ticket_id          -> service_tickets.ticket_number
  psa_tickets.title              -> service_tickets.title
  psa_tickets.client_id          -> service_tickets.client_id
  psa_tickets.client_name        -> service_tickets.client_name
  psa_tickets.type               -> service_tickets.category
  psa_tickets.priority (UPPER)   -> service_tickets.priority (lower)
  psa_tickets.status  (UPPER)    -> service_tickets.status  (lower)
  psa_tickets.technician         -> service_tickets.assigned_to
  psa_tickets.resolution_note    -> service_tickets.engineer_notes
  psa_tickets.created_date       -> service_tickets.created_at
  psa_tickets.resolved_date      -> service_tickets.resolved_at
  (constructed from title+type)  -> service_tickets.description

Safe to re-run: uses upsert on ticket_number, so existing rows are updated
rather than duplicated.

Usage:
    python scripts/backfill_service_tickets.py

Requires SUPABASE_URL and SUPABASE_SERVICE_KEY in .env
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("[ERROR] SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
    sys.exit(1)

from supabase import create_client
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

BATCH_SIZE = 50

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_status(psa_status: str) -> str:
    """Uppercase PSA status -> lowercase service_tickets status."""
    return {
        "OPEN":           "open",
        "IN_PROGRESS":    "in_progress",
        "WAITING_CLIENT": "waiting_client",
        "RESOLVED":       "resolved",
        "CLOSED":         "closed",
    }.get(psa_status.upper(), psa_status.lower())


def _map_priority(psa_priority: str) -> str:
    """Uppercase PSA priority -> lowercase service_tickets priority."""
    return psa_priority.lower() if psa_priority else "medium"


# Allowed values for the service_tickets.category CHECK constraint:
#   network, hardware, software, security, email, cloud, backup, other
_SECURITY_KW  = {"ransomware","malware","phishing","edr","threat","vulnerability",
                  "ssl","certificate","firewall","vpn","password","intrusion","mfa","2fa"}
_NETWORK_KW   = {"wi-fi","wifi","network","switch","vlan","dns","dhcp","routing",
                  "bandwidth","latency","port","sfp","ap ","access point","internet"}
_EMAIL_KW     = {"email","outlook","exchange","smtp","mail"}
_CLOUD_KW     = {"azure","aws","office 365","o365","microsoft 365","m365","saas",
                  "subscription","cloud","onedrive","sharepoint online","teams"}
_BACKUP_KW    = {"backup","restore","recovery","disaster","snapshot","replication"}
_HARDWARE_KW  = {"printer","server","laptop","desktop","monitor","ups","battery",
                  "disk","drive","ram","cpu","hardware","workstation","device",
                  "cable","port flapping","sfp module"}
_SOFTWARE_KW  = {"software","application","app","driver","update","patch","install",
                  "sharepoint","teams","database","sql","erp","crm","script",
                  "permissions","policy","group policy","active directory","ad "}


def _map_category(ticket_type: str, title: str) -> str:
    """
    Map PSA ticket type + title to the allowed service_tickets category values.

    Allowed: network, hardware, software, security, email, cloud, backup, other
    """
    t = title.lower()
    if any(k in t for k in _SECURITY_KW):  return "security"
    if any(k in t for k in _NETWORK_KW):   return "network"
    if any(k in t for k in _EMAIL_KW):     return "email"
    if any(k in t for k in _CLOUD_KW):     return "cloud"
    if any(k in t for k in _BACKUP_KW):    return "backup"
    if any(k in t for k in _HARDWARE_KW):  return "hardware"
    if any(k in t for k in _SOFTWARE_KW):  return "software"
    return "other"


def _build_description(row: dict) -> str:
    """Construct a description field from PSA fields."""
    parts = []
    if row.get("type"):
        parts.append(f"Type: {row['type']}")
    if row.get("resolution_note"):
        parts.append(f"Resolution: {row['resolution_note']}")
    if row.get("hours_billed"):
        parts.append(f"Hours billed: {row['hours_billed']}h")
    if row.get("sla_met") is not None:
        parts.append(f"SLA met: {'Yes' if row['sla_met'] else 'No'}")
    return "\n".join(parts) if parts else row.get("title", "")


def _to_timestamp(date_str: str | None) -> str | None:
    """Convert a date string '2025-10-18' to ISO timestamp '2025-10-18T00:00:00+00:00'."""
    if not date_str:
        return None
    # Already a timestamp?
    if "T" in date_str:
        return date_str
    return date_str + "T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Red Key Sandbox MSP -- Backfill service_tickets from psa_tickets")
    print("=" * 60)

    # 1. Check current state
    existing = sb.table("service_tickets").select("ticket_number", count="exact").execute()
    existing_count = existing.count or 0
    existing_numbers = {r["ticket_number"] for r in existing.data} if existing.data else set()
    print(f"\nservice_tickets currently: {existing_count} rows")

    # 2. Fetch all psa_tickets
    print("Fetching psa_tickets...")
    all_psa: list[dict] = []
    offset = 0
    while True:
        res = sb.table("psa_tickets").select("*").range(offset, offset + 999).execute()
        all_psa.extend(res.data)
        if len(res.data) < 1000:
            break
        offset += 1000

    print(f"psa_tickets fetched: {len(all_psa)} rows")

    # 3. Build service_tickets rows
    rows_to_upsert: list[dict] = []
    for r in all_psa:
        row = {
            "ticket_number":  r["ticket_id"],           # e.g. TKT-2025-5001
            "client_id":      r["client_id"],
            "client_name":    r.get("client_name", ""),
            "title":          r.get("title", ""),
            "description":    _build_description(r),
            "category":       _map_category(r.get("type", ""), r.get("title", "")),
            "priority":       _map_priority(r.get("priority", "medium")),
            "status":         _map_status(r.get("status", "closed")),
            "assigned_to":    r.get("technician") or None,
            "engineer_notes": r.get("resolution_note") or None,
            "created_at":     _to_timestamp(r.get("created_date")),
            "resolved_at":    _to_timestamp(r.get("resolved_date")),
        }
        rows_to_upsert.append(row)

    print(f"Rows prepared for upsert: {len(rows_to_upsert)}")

    # 4. Upsert in batches
    inserted = 0
    failed   = 0
    for i in range(0, len(rows_to_upsert), BATCH_SIZE):
        batch = rows_to_upsert[i : i + BATCH_SIZE]
        try:
            sb.table("service_tickets").upsert(
                batch, on_conflict="ticket_number"
            ).execute()
            inserted += len(batch)
        except Exception as exc:
            failed += len(batch)
            print(f"\n  [ERROR] Batch {i // BATCH_SIZE + 1} failed: {exc}")
        print(f"  Progress: {inserted + failed}/{len(rows_to_upsert)} "
              f"({inserted} ok, {failed} failed)", end="\r")

    print()
    print()
    print("=" * 60)
    print(f"  Upserted: {inserted}")
    print(f"  Failed:   {failed}")

    # 5. Verify final state
    final = sb.table("service_tickets").select("status", count="exact").execute()
    from collections import Counter
    by_status = Counter(r["status"] for r in (final.data or []))
    print(f"  Final service_tickets count: {final.count}")
    print(f"  By status: {dict(by_status)}")
    print("=" * 60)

    if failed == 0:
        print("\nDone. Refresh the Admin Dashboard -- ticket counts will now be populated.")
    else:
        print(f"\nCompleted with {failed} errors. Check output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
