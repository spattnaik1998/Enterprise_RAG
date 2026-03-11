#!/usr/bin/env python
"""
Portal Credential Setup
-----------------------
Generates login credentials for the Red Key Sandbox Client Portal:
  - 1 MSP admin account (username: msp_admin)
  - 50 client accounts  (one per client in data/enterprise/clients.json)

Actions performed:
  1. Generates secure random passwords
  2. Hashes each password with bcrypt
  3. Upserts rows into Supabase `portal_users` table
  4. Writes PLAINTEXT credentials to data/portal_credentials.json
     (gitignored -- keep this file secret)

Prerequisites:
  - Run scripts/setup_portal_tables.sql in Supabase SQL Editor first
  - SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env
  - data/enterprise/clients.json must exist

Usage:
    python scripts/setup_portal.py           # create/update all accounts
    python scripts/setup_portal.py --reset   # delete ALL portal_users first
"""
from __future__ import annotations

import json
import os
import secrets
import string
import sys
from pathlib import Path

# Windows cp1252 fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Allow running from project root or scripts/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from supabase import create_client

# Import auth helpers for hashing
from app.auth import hash_password

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MSP_USERNAME = "msp_admin"
MSP_PASSWORD = "RedKeySandbox@2025!"          # fixed, easy to remember
MSP_DISPLAY  = "Red Key Sandbox MSP Admin"

CLIENTS_FILE = ROOT / "data" / "enterprise" / "clients.json"
CREDS_FILE   = ROOT / "data" / "portal_credentials.json"

_CHARS = string.ascii_letters + string.digits + "!@#$%"


def _gen_password(length: int = 12) -> str:
    """Generate a cryptographically secure random password."""
    while True:
        pwd = "".join(secrets.choice(_CHARS) for _ in range(length))
        # Ensure at least one of each required character class
        if (
            any(c.isupper() for c in pwd)
            and any(c.islower() for c in pwd)
            and any(c.isdigit() for c in pwd)
            and any(c in "!@#$%" for c in pwd)
        ):
            return pwd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(reset: bool = False) -> None:
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY", "")
    if not supabase_url or not supabase_key:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
        sys.exit(1)

    db = create_client(supabase_url, supabase_key)

    # Load clients
    if not CLIENTS_FILE.exists():
        print(f"ERROR: {CLIENTS_FILE} not found. Run generate_enterprise_data.py first.")
        sys.exit(1)

    with CLIENTS_FILE.open(encoding="utf-8") as f:
        raw = json.load(f)
    clients: list[dict] = raw.get("records", raw) if isinstance(raw, dict) else raw
    print(f"Loaded {len(clients)} clients from {CLIENTS_FILE.name}")

    # Optionally wipe existing portal_users
    if reset:
        confirm = input("WARNING: This will delete ALL portal_users. Type YES to confirm: ")
        if confirm.strip() != "YES":
            print("Aborted.")
            sys.exit(0)
        db.table("portal_users").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        print("Existing portal_users deleted.")

    # Build user records
    credentials_plain: list[dict] = []
    rows_to_upsert: list[dict] = []

    # MSP admin
    msp_hash = hash_password(MSP_PASSWORD)
    rows_to_upsert.append({
        "username":      MSP_USERNAME,
        "password_hash": msp_hash,
        "role":          "msp",
        "client_id":     None,
        "client_name":   None,
    })
    credentials_plain.append({
        "role":     "msp",
        "username": MSP_USERNAME,
        "password": MSP_PASSWORD,
        "note":     "MSP admin — access to RAG, forecast, logs, and all tickets",
    })

    # Client accounts
    for client in clients:
        client_id   = client.get("id", "")
        client_name = client.get("name", client_id)
        username    = client_id.lower()          # e.g. "clt-001"
        password    = _gen_password()
        pwd_hash    = hash_password(password)

        rows_to_upsert.append({
            "username":      username,
            "password_hash": pwd_hash,
            "role":          "client",
            "client_id":     client_id,
            "client_name":   client_name,
        })
        credentials_plain.append({
            "role":        "client",
            "client_id":   client_id,
            "client_name": client_name,
            "username":    username,
            "password":    password,
            "industry":    client.get("industry", ""),
            "domain":      client.get("domain", ""),
        })

    print(f"Generated credentials for {len(rows_to_upsert)} accounts (1 MSP + {len(clients)} clients)")

    # Upsert into Supabase (on_conflict=username -> update password_hash)
    print("Upserting into Supabase portal_users...")
    batch_size = 20
    for i in range(0, len(rows_to_upsert), batch_size):
        batch = rows_to_upsert[i : i + batch_size]
        db.table("portal_users").upsert(batch, on_conflict="username").execute()
        print(f"  Upserted rows {i+1}-{min(i+batch_size, len(rows_to_upsert))}")

    print("Supabase upsert complete.")

    # Write plaintext credentials to JSON (keep this file secret)
    CREDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CREDS_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "warning": "KEEP THIS FILE SECRET -- contains plaintext passwords",
                "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                "msp_portal_url": "http://localhost:8000",
                "credentials": credentials_plain,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Plaintext credentials written to {CREDS_FILE}")
    print()
    print("=" * 60)
    print("MSP ADMIN CREDENTIALS")
    print("=" * 60)
    print(f"  URL      : http://localhost:8000")
    print(f"  Username : {MSP_USERNAME}")
    print(f"  Password : {MSP_PASSWORD}")
    print("=" * 60)
    print(f"Client credentials saved to: {CREDS_FILE}")
    print("Setup complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Set up portal credentials")
    parser.add_argument("--reset", action="store_true", help="Delete existing portal_users first")
    args = parser.parse_args()
    main(reset=args.reset)
