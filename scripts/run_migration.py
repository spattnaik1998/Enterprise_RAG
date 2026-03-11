#!/usr/bin/env python3
"""
Run Supabase SQL migrations.
Since the Supabase Python SDK doesn't support direct SQL execution,
this script provides instructions for manual execution in the dashboard.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

def run_migration(migration_file: str):
    """Display SQL migration for manual execution."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
        return False

    if not os.path.exists(migration_file):
        print(f"ERROR: Migration file not found: {migration_file}")
        return False

    with open(migration_file, 'r') as f:
        sql = f.read().strip()

    print("\n" + "="*70)
    print("SUPABASE SQL MIGRATION")
    print("="*70)
    print(f"File: {migration_file}\n")
    print("SQL to execute:")
    print("-"*70)
    print(sql)
    print("-"*70)
    print("\nINSTRUCTIONS:")
    print("1. Go to https://supabase.com/dashboard")
    print("2. Select your project")
    print("3. Click 'SQL Editor' in the left sidebar")
    print("4. Click 'New Query'")
    print("5. Paste the above SQL")
    print("6. Click 'Run'")
    print("\nAlternatively, use the Supabase CLI:")
    print(f"  npx supabase db push --db-url '{url.replace('https://', 'postgresql://')}' < {migration_file}")
    print("\nAfter running the migration, run the setup script:")
    print("  python scripts/setup_engineer_credentials.py")
    print("="*70 + "\n")

    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_migration.py <migration_file>")
        print("Example: python scripts/run_migration.py supabase/migrations/20260311_add_engineer_role.sql")
        sys.exit(1)

    migration_file = sys.argv[1]
    success = run_migration(migration_file)
    sys.exit(0 if success else 1)
