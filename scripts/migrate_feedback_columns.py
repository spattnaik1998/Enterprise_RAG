"""
Migration: Add user feedback columns to chat_logs table.

Adds:
  - user_rating (int, nullable, 1-5 stars)
  - user_feedback (text, nullable, free-form comment)

Run once:
    python scripts/migrate_feedback_columns.py
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

    print("Adding user_rating and user_feedback columns to chat_logs...")
    print()
    print("Run this SQL in the Supabase SQL Editor:")
    print()
    print("ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS user_rating INTEGER;")
    print("ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS user_feedback TEXT;")
    print()
    print("-- Optional: add a check constraint for valid ratings")
    print("ALTER TABLE chat_logs ADD CONSTRAINT check_rating")
    print("  CHECK (user_rating IS NULL OR (user_rating >= 1 AND user_rating <= 5));")
    print()
    print("Done. Columns are nullable so existing rows are unaffected.")


if __name__ == "__main__":
    main()
