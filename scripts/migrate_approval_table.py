"""
Migrate Approval Table
---------------------
Create approval_requests table in Supabase (idempotent).
"""
import os
import sys

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def main():
    """Create approval_requests table."""
    try:
        from supabase import create_client

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

        if not supabase_url or not supabase_key:
            logger.error("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
            return False

        sb = create_client(supabase_url, supabase_key)
        logger.info("[Migration] Connecting to Supabase...")

        # Try RPC approach first (if exec_sql exists)
        sql = """
        CREATE TABLE IF NOT EXISTS approval_requests (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            action TEXT NOT NULL,
            username TEXT NOT NULL,
            justification TEXT,
            status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
            approver TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            resolved_at TIMESTAMP WITH TIME ZONE
        );
        
        CREATE INDEX IF NOT EXISTS idx_approval_status ON approval_requests(status);
        CREATE INDEX IF NOT EXISTS idx_approval_username ON approval_requests(username);
        """

        try:
            # Try executing via RPC (if available)
            result = sb.rpc("exec_sql", {"sql": sql}).execute()
            logger.info("[Migration] approval_requests table created via RPC")
            return True
        except Exception as e:
            logger.warning(f"[Migration] RPC approach failed: {e}")
            logger.info("[Migration] Manual SQL creation required:")
            logger.info(sql)
            logger.info("")
            logger.info("Run the above SQL in your Supabase SQL editor to create the table.")
            return False

    except Exception as e:
        logger.error(f"[Migration] Error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
