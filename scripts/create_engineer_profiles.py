"""
create_engineer_profiles.py
----------------------------
Extracts all unique engineer names from the psa_tickets table in Supabase,
builds profile rows, and upserts them into the `engineers` table.

Engineer email format: firstname.lastname@redkeysandbox.com

Profile fields:
  name              -- full name (from psa_tickets.technician)
  email             -- firstname.lastname@redkeysandbox.com
  total_tickets     -- total tickets assigned
  resolved_tickets  -- tickets with status RESOLVED or CLOSED
  avg_hours_per_ticket -- average hours billed per ticket
  sla_rate          -- SLA met % (0.0 - 100.0)
  primary_specialty -- most common ticket type worked
  ticket_type_breakdown -- JSON object with counts per type

Usage:
    python scripts/create_engineer_profiles.py

Requires SUPABASE_URL and SUPABASE_SERVICE_KEY in .env
"""
from __future__ import annotations

import json
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


def _make_email(full_name: str) -> str:
    """Convert 'First Last' -> 'first.last@redkeysandbox.com'."""
    parts = full_name.lower().split()
    return ".".join(parts) + "@redkeysandbox.com"


def _fetch_all_psa_tickets() -> list[dict]:
    """Page through all psa_tickets rows from Supabase."""
    all_rows: list[dict] = []
    offset = 0
    while True:
        res = sb.table("psa_tickets").select(
            "technician,status,sla_met,hours_billed,type"
        ).range(offset, offset + 999).execute()
        all_rows.extend(res.data)
        if len(res.data) < 1000:
            break
        offset += 1000
    return all_rows


def _aggregate_engineers(tickets: list[dict]) -> list[dict]:
    """Aggregate per-engineer stats from ticket rows."""
    techs: dict[str, dict] = {}

    for r in tickets:
        name = r.get("technician")
        if not name:
            continue
        if name not in techs:
            techs[name] = {
                "total": 0,
                "resolved": 0,
                "sla_met": 0,
                "sla_total": 0,
                "hours": 0.0,
                "categories": {},
            }
        t = techs[name]
        t["total"] += 1
        if r.get("status") in ("RESOLVED", "CLOSED"):
            t["resolved"] += 1
        if r.get("sla_met") is True:
            t["sla_met"] += 1
        if r.get("sla_met") is not None:
            t["sla_total"] += 1
        t["hours"] += float(r.get("hours_billed") or 0)
        cat = r.get("type") or "other"
        t["categories"][cat] = t["categories"].get(cat, 0) + 1

    rows = []
    for name, info in sorted(techs.items()):
        top_cat = max(info["categories"], key=info["categories"].get)
        sla_rate = round(info["sla_met"] / info["sla_total"] * 100, 1) if info["sla_total"] else 0.0
        avg_hours = round(info["hours"] / info["total"], 2) if info["total"] else 0.0
        rows.append({
            "name": name,
            "email": _make_email(name),
            "total_tickets": info["total"],
            "resolved_tickets": info["resolved"],
            "avg_hours_per_ticket": avg_hours,
            "sla_rate": sla_rate,
            "primary_specialty": top_cat,
            "ticket_type_breakdown": info["categories"],
        })

    return rows


def _ensure_table_exists() -> bool:
    """
    Check if engineers table exists by doing a lightweight select.
    Returns True if table exists, False otherwise.
    """
    try:
        sb.table("engineers").select("name").limit(1).execute()
        return True
    except Exception as exc:
        if "relation" in str(exc).lower() or "does not exist" in str(exc).lower() or "42P01" in str(exc):
            return False
        # Other error — re-raise
        raise


def main() -> None:
    print("=" * 60)
    print("Red Key Sandbox MSP -- Create Engineer Profiles")
    print("=" * 60)

    # 1. Fetch tickets from Supabase
    print("\nFetching psa_tickets from Supabase...")
    tickets = _fetch_all_psa_tickets()
    print(f"  Fetched {len(tickets)} ticket rows")

    # 2. Aggregate
    engineer_rows = _aggregate_engineers(tickets)
    print(f"  Found {len(engineer_rows)} unique engineers")
    print()
    for row in engineer_rows:
        print(f"  {row['name']}")
        print(f"    email:    {row['email']}")
        print(f"    tickets:  {row['total_tickets']} total, {row['resolved_tickets']} resolved")
        print(f"    sla_rate: {row['sla_rate']}%")
        print(f"    avg_hrs:  {row['avg_hours_per_ticket']}h/ticket")
        print(f"    specialty:{row['primary_specialty']}")
        print(f"    breakdown:{row['ticket_type_breakdown']}")
        print()

    # 3. Check table exists
    print("Checking engineers table...")
    if not _ensure_table_exists():
        print("[WARN] engineers table not found in Supabase.")
        print("       Please create it using the SQL below, then re-run this script.\n")
        print("-" * 60)
        print("""
CREATE TABLE IF NOT EXISTS engineers (
    id                    BIGSERIAL PRIMARY KEY,
    name                  TEXT NOT NULL UNIQUE,
    email                 TEXT NOT NULL UNIQUE,
    total_tickets         INTEGER DEFAULT 0,
    resolved_tickets      INTEGER DEFAULT 0,
    avg_hours_per_ticket  NUMERIC(6,2) DEFAULT 0,
    sla_rate              NUMERIC(5,1) DEFAULT 0,
    primary_specialty     TEXT DEFAULT 'other',
    ticket_type_breakdown JSONB DEFAULT '{}',
    created_at            TIMESTAMPTZ DEFAULT now(),
    updated_at            TIMESTAMPTZ DEFAULT now()
);
        """.strip())
        print("-" * 60)
        sys.exit(1)

    print("  engineers table found.")

    # 4. Upsert
    print(f"\nUpserting {len(engineer_rows)} engineer profiles...")
    try:
        sb.table("engineers").upsert(
            engineer_rows,
            on_conflict="name",
        ).execute()
        print(f"  Upserted {len(engineer_rows)} rows successfully.")
    except Exception as exc:
        print(f"[ERROR] Upsert failed: {exc}")
        sys.exit(1)

    # 5. Verify
    print("\nVerifying stored profiles...")
    final = sb.table("engineers").select("name,email,total_tickets,sla_rate,primary_specialty").execute()
    print(f"  engineers table now has {len(final.data)} rows:\n")
    for row in final.data:
        print(f"  {row['name']:<20} {row['email']:<45} "
              f"tickets={row['total_tickets']:<4} sla={row['sla_rate']}% specialty={row['primary_specialty']}")

    print()
    print("=" * 60)
    print("Done. Engineer profiles are live in Supabase.")
    print("=" * 60)


if __name__ == "__main__":
    main()
