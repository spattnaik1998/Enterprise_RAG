#!/usr/bin/env python3
"""
Execute Supabase SQL migrations via REST API.
Uses the Supabase SQL API endpoint for direct SQL execution.
"""

import sys
import os
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

def execute_migration(migration_file: str):
    """Execute SQL migration via Supabase PostgreSQL connection."""
    url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not service_key:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
        return False

    if not os.path.exists(migration_file):
        print(f"ERROR: Migration file not found: {migration_file}")
        return False

    with open(migration_file, 'r') as f:
        sql = f.read().strip()

    print(f"Executing migration: {migration_file}")
    print(f"SQL Preview:\n{sql[:200]}...\n")

    try:
        # Try using psycopg3 to connect directly to Supabase PostgreSQL
        import psycopg

        # Extract connection info from SUPABASE_URL
        # Format: https://project-ref.supabase.co
        # PostgreSQL connection: postgresql://postgres:password@project-ref.supabase.co:5432/postgres
        project_ref = url.replace("https://", "").split(".supabase")[0]
        password = service_key  # The service_key can be used as the password

        db_url = f"postgresql://postgres:{service_key}@{project_ref}.supabase.co:5432/postgres"

        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                conn.commit()

        print("✓ Migration executed successfully!")
        print("\nNext steps:")
        print("  1. python scripts/setup_engineer_credentials.py  # Create engineer accounts")
        print("  2. uvicorn app.server:app --reload --port 8000   # Start the app")
        return True

    except ImportError:
        print("psycopg3 not installed. Trying alternative method...\n")
        return execute_via_api(migration_file, url, service_key, sql)
    except Exception as e:
        print(f"✗ Error executing migration: {e}")
        return False

def execute_via_api(migration_file: str, url: str, service_key: str, sql: str):
    """Try executing via Supabase REST API."""
    try:
        import requests

        # Try the internal SQL execution endpoint
        api_url = f"{url}/rest/v1/rpc/execute_sql"
        headers = {
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json",
            "apikey": service_key,
        }

        payload = {"query": sql}

        response = requests.post(api_url, json=payload, headers=headers, timeout=30)

        if response.status_code in (200, 204):
            print("✓ Migration executed via REST API successfully!")
            return True
        else:
            print(f"REST API Error ({response.status_code}): {response.text}")
            return show_manual_instructions(sql)
    except Exception as e:
        print(f"REST API Error: {e}")
        return show_manual_instructions(sql)

def show_manual_instructions(sql: str):
    """Show instructions for manual execution."""
    print("\n" + "="*70)
    print("AUTOMATIC EXECUTION FAILED - MANUAL EXECUTION REQUIRED")
    print("="*70)
    print("\nPlease run this SQL manually in your Supabase dashboard:\n")
    print(sql)
    print("\n" + "="*70)
    print("\nManual Steps:")
    print("1. Go to https://supabase.com/dashboard")
    print("2. Select your project")
    print("3. Click 'SQL Editor' in the left sidebar")
    print("4. Click 'New Query'")
    print("5. Copy-paste the above SQL")
    print("6. Click 'Run'")
    print("\nAfter manual execution, run:")
    print("  python scripts/setup_engineer_credentials.py")
    print("="*70 + "\n")
    return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to the engineer role migration
        migration_file = "supabase/migrations/20260311_add_engineer_role.sql"
    else:
        migration_file = sys.argv[1]

    success = execute_migration(migration_file)
    sys.exit(0 if success else 1)
