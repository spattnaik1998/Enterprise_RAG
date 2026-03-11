"""
setup_engineer_credentials.py
-----------------------------
Creates login accounts for all engineers in the engineers table.

Each engineer gets:
  - username: firstname_lastname (lowercase)
  - password: auto-generated secure random
  - role: engineer
  - email: from engineers table

Inserts into portal_users table.

Usage:
    python scripts/setup_engineer_credentials.py

Requires SUPABASE_URL and SUPABASE_SERVICE_KEY in .env
"""
from __future__ import annotations

import json
import os
import secrets
import string
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from supabase import create_client
from app.auth import hash_password

CREDS_FILE = ROOT / "data" / "portal_credentials.json"
_CHARS = string.ascii_letters + string.digits + "!@#$%"


def _gen_password(length: int = 12) -> str:
    """Generate a cryptographically secure random password."""
    while True:
        pwd = "".join(secrets.choice(_CHARS) for _ in range(length))
        if (
            any(c.isupper() for c in pwd)
            and any(c.islower() for c in pwd)
            and any(c.isdigit() for c in pwd)
            and any(c in "!@#$%" for c in pwd)
        ):
            return pwd


def _username_from_email(email: str) -> str:
    """Extract firstname_lastname from email (e.g. alex.rivera@... -> alex_rivera)."""
    local = email.split("@")[0].lower()  # alex.rivera
    return local.replace(".", "_")  # alex_rivera


def main() -> None:
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY", "")
    if not supabase_url or not supabase_key:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
        sys.exit(1)

    db = create_client(supabase_url, supabase_key)

    print("=" * 60)
    print("Red Key Sandbox -- Engineer Portal Credentials Setup")
    print("=" * 60)

    # Fetch all engineers
    print("\nFetching engineers from engineers table...")
    try:
        res = db.table("engineers").select("*").order("total_tickets", desc=True).execute()
        engineers = res.data or []
    except Exception as exc:
        print(f"[ERROR] Failed to fetch engineers: {exc}")
        sys.exit(1)

    if not engineers:
        print("[ERROR] No engineers found in engineers table")
        sys.exit(1)

    print(f"Found {len(engineers)} engineers")

    # Build engineer user records
    rows_to_upsert: list[dict] = []
    credentials_plain: list[dict] = []

    for eng in engineers:
        name = eng.get("name", "")
        email = eng.get("email", "")
        if not name or not email:
            print(f"[WARN] Skipping engineer with missing name/email: {eng}")
            continue

        username = _username_from_email(email)
        password = _gen_password()
        pwd_hash = hash_password(password)

        rows_to_upsert.append({
            "username": username,
            "password_hash": pwd_hash,
            "role": "engineer",
            "client_id": None,
            "client_name": None,
        })
        credentials_plain.append({
            "role": "engineer",
            "name": name,
            "email": email,
            "username": username,
            "password": password,
        })

    print(f"\nGenerated credentials for {len(rows_to_upsert)} engineers")

    # Upsert into Supabase
    print("Upserting engineer accounts into Supabase portal_users...")
    try:
        for i in range(0, len(rows_to_upsert), 10):
            batch = rows_to_upsert[i : i + 10]
            db.table("portal_users").upsert(batch, on_conflict="username").execute()
            print(f"  Upserted rows {i+1}-{min(i+10, len(rows_to_upsert))}")
    except Exception as exc:
        print(f"[ERROR] Upsert failed: {exc}")
        sys.exit(1)

    print("Supabase upsert complete.")

    # Update credentials file to include engineer credentials
    if CREDS_FILE.exists():
        with CREDS_FILE.open("r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {
            "warning": "KEEP THIS FILE SECRET -- contains plaintext passwords",
            "msp_portal_url": "http://localhost:8000",
            "credentials": [],
        }

    # Filter out old engineer entries, then add new ones
    existing["credentials"] = [c for c in existing.get("credentials", []) if c.get("role") != "engineer"]
    existing["credentials"].extend(credentials_plain)

    CREDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CREDS_FILE.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"\nCredentials updated in {CREDS_FILE}")
    print()
    print("=" * 60)
    print("ENGINEER PORTAL CREDENTIALS")
    print("=" * 60)
    for eng in credentials_plain:
        print(f"\n{eng['name']}")
        print(f"  Email:    {eng['email']}")
        print(f"  Username: {eng['username']}")
        print(f"  Password: {eng['password']}")
    print()
    print("=" * 60)
    print("Setup complete. Engineers can now log in at: http://localhost:8000/engineer")
    print("=" * 60)


if __name__ == "__main__":
    main()
