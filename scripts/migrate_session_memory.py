"""
Migration: Add answer_preview column to chat_logs table.

Run once:
    python scripts/migrate_session_memory.py
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
load_dotenv()


def main() -> None:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
        sys.exit(1)

    from supabase import create_client
    sb = create_client(url, key)

    # Add answer_preview column (first 300 chars of answer text)
    # This is idempotent -- Supabase will skip if column already exists
    sql = """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'chat_logs' AND column_name = 'answer_preview'
        ) THEN
            ALTER TABLE chat_logs ADD COLUMN answer_preview TEXT;
        END IF;
    END $$;
    """
    try:
        sb.rpc("exec_sql", {"sql": sql}).execute()
        print("OK: answer_preview column added (or already exists)")
    except Exception as exc:
        # If exec_sql RPC doesn't exist, print the SQL for manual execution
        print(f"Auto-migration failed: {exc}")
        print("\nRun this SQL manually in the Supabase SQL Editor:\n")
        print("ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS answer_preview TEXT;")
        print("\n(PostgreSQL supports IF NOT EXISTS on ADD COLUMN)")


if __name__ == "__main__":
    main()
