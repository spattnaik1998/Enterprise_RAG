"""
migrate_to_supabase.py
-----------------------
One-time migration of all local data to Supabase.

Migrates (in dependency order):
  1. clients           <- data/enterprise/clients.json
  2. invoices          <- data/enterprise/invoices.json
  3. invoice_line_items <- (normalised from invoices.json)
  4. psa_tickets       <- data/enterprise/psa_tickets.json
  5. crm_profiles      <- data/enterprise/crm_profiles.json
  6. communications    <- data/enterprise/communications.json
  7. contracts         <- data/enterprise/contracts.json
  8. rag_chunks        <- data/index/chunks.json + data/index/faiss.index
                         (embeddings extracted via faiss.reconstruct)
  9. chat_logs         <- data/chat_logs/chat_history.jsonl (if present)

Usage:
  python scripts/migrate_to_supabase.py

Requires in .env:
  SUPABASE_URL=https://xxxx.supabase.co
  SUPABASE_SERVICE_KEY=eyJ...    (service_role key from Settings > API)

Run Phase II (python -m src.main phase2) before running this script
so that data/index/ files exist.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Encoding fix for Windows cp1252 terminals
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR     = Path("data/enterprise")
INDEX_DIR    = Path("data/index")
CHAT_LOG     = Path("data/chat_logs/chat_history.jsonl")
BATCH_SIZE   = 50          # rows per Supabase insert call

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_env() -> None:
    missing = [k for k, v in [("SUPABASE_URL", SUPABASE_URL), ("SUPABASE_SERVICE_KEY", SUPABASE_KEY)] if not v]
    if missing:
        print(f"[ERROR] Missing env vars: {', '.join(missing)}")
        print("        Add them to your .env and re-run.")
        sys.exit(1)


def _client():
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _load_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    if isinstance(data, list):
        return data
    return [data]


def _batch_insert(sb, table: str, rows: list[dict], label: str) -> None:
    total = len(rows)
    inserted = 0
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        sb.table(table).insert(batch).execute()
        inserted += len(batch)
        print(f"  [{label}] {inserted}/{total} rows inserted", end="\r")
    print(f"  [{label}] {total} rows inserted.        ")


# ---------------------------------------------------------------------------
# Per-table migrations
# ---------------------------------------------------------------------------

def migrate_clients(sb) -> None:
    print("[1/9] Migrating clients...")
    records = _load_json(DATA_DIR / "clients.json")
    rows = []
    for r in records:
        rows.append({
            "client_id":             r["id"],
            "name":                  r["name"],
            "industry":              r.get("industry", ""),
            "size":                  r.get("size", ""),
            "users":                 r.get("users", 0),
            "services":              r.get("services", []),
            "reliability":           r.get("reliability", ""),
            "payment_terms":         r.get("terms", ""),
            "start_date":            r.get("start"),
            "has_projects":          r.get("has_projects", False),
            "account_manager":       r.get("manager", ""),
            "domain":                r.get("domain", ""),
            "billing_contact_email": r.get("billing_contact_email", ""),
            "address":               r.get("address", {}),
            "contacts":              r.get("contacts", {}),
        })
    _batch_insert(sb, "clients", rows, "clients")


def migrate_invoices(sb) -> None:
    print("[2/9] Migrating invoices...")
    records = _load_json(DATA_DIR / "invoices.json")
    invoice_rows = []
    line_item_rows = []

    for r in records:
        invoice_rows.append({
            "invoice_id":         r["invoice_id"],
            "client_id":          r["client_id"],
            "client_name":        r.get("client_name", ""),
            "billing_contact":    r.get("billing_contact", ""),
            "account_manager":    r.get("account_manager", ""),
            "billing_period":     r.get("billing_period", ""),
            "invoice_type":       r.get("invoice_type", ""),
            "invoice_date":       r["invoice_date"],
            "due_date":           r["due_date"],
            "payment_terms":      r.get("payment_terms", ""),
            "status":             r.get("status", ""),
            "days_outstanding":   r.get("days_outstanding", 0),
            "subtotal":           r.get("subtotal", 0.0),
            "tax_rate":           r.get("tax_rate", 0.0),
            "tax_amount":         r.get("tax_amount", 0.0),
            "total_amount":       r.get("total_amount", 0.0),
            "amount_paid":        r.get("amount_paid", 0.0),
            "balance_due":        r.get("balance_due", 0.0),
            "currency":           r.get("currency", "USD"),
            "late_fee_applicable": r.get("late_fee_applicable", False),
            "late_fee_amount":    r.get("late_fee_amount", 0.0),
            "notes":              r.get("notes", ""),
        })

        for li in r.get("line_items", []):
            line_item_rows.append({
                "invoice_id":   r["invoice_id"],
                "client_id":    r["client_id"],
                "service_code": li.get("service_code", ""),
                "description":  li.get("description", ""),
                "quantity":     int(li.get("quantity", 1)),
                "unit_price":   float(li.get("unit_price", 0.0)),
                "amount":       float(li.get("amount", 0.0)),
            })

    _batch_insert(sb, "invoices", invoice_rows, "invoices")

    print("[3/9] Migrating invoice line items...")
    _batch_insert(sb, "invoice_line_items", line_item_rows, "invoice_line_items")


def migrate_psa_tickets(sb) -> None:
    print("[4/9] Migrating PSA tickets...")
    records = _load_json(DATA_DIR / "psa_tickets.json")
    rows = []
    for r in records:
        rows.append({
            "ticket_id":      r["ticket_id"],
            "client_id":      r["client_id"],
            "client_name":    r.get("client_name", ""),
            "linked_invoice": r.get("linked_invoice"),
            "type":           r.get("type", ""),
            "priority":       r.get("priority", ""),
            "title":          r.get("title", ""),
            "created_date":   r["created_date"],
            "resolved_date":  r.get("resolved_date"),
            "technician":     r.get("technician", ""),
            "hours_billed":   float(r.get("hours_billed", 0.0)),
            "billing_status": r.get("billing_status", ""),
            "status":         r.get("status", ""),
            "sla_met":        r.get("sla_met"),
            "resolution_note": r.get("resolution_note", ""),
        })
    _batch_insert(sb, "psa_tickets", rows, "psa_tickets")


def migrate_crm_profiles(sb) -> None:
    print("[5/9] Migrating CRM profiles...")
    records = _load_json(DATA_DIR / "crm_profiles.json")
    rows = []
    for r in records:
        rows.append({
            "client_id":             r["client_id"],
            "client_name":           r.get("client_name", ""),
            "industry":              r.get("industry", ""),
            "employee_count":        r.get("employee_count", 0),
            "client_since":          r.get("client_since"),
            "years_as_client":       float(r.get("years_as_client", 0.0)),
            "account_manager":       r.get("account_manager", ""),
            "account_health":        r.get("account_health", ""),
            "health_note":           r.get("health_note", ""),
            "payment_reliability":   r.get("payment_reliability", ""),
            "total_billed_ytd":      float(r.get("total_billed_ytd", 0.0)),
            "total_paid_ytd":        float(r.get("total_paid_ytd", 0.0)),
            "open_balance":          float(r.get("open_balance", 0.0)),
            "overdue_invoice_count": int(r.get("overdue_invoice_count", 0)),
            "overdue_balance":       float(r.get("overdue_balance", 0.0)),
            "upsell_opportunities":  r.get("upsell_opportunities", []),
            "last_qbr_date":         r.get("last_qbr_date"),
            "nps_score":             r.get("nps_score"),
            "contacts":              r.get("contacts", {}),
            "address":               r.get("address", {}),
        })
    _batch_insert(sb, "crm_profiles", rows, "crm_profiles")


def migrate_communications(sb) -> None:
    print("[6/9] Migrating communications...")
    records = _load_json(DATA_DIR / "communications.json")
    rows = []
    for r in records:
        rows.append({
            "comm_id":              r["comm_id"],
            "invoice_id":           r["invoice_id"],
            "client_id":            r["client_id"],
            "client_name":          r.get("client_name", ""),
            "sent_date":            r["sent_date"],
            "reminder_sequence":    int(r.get("reminder_sequence", 1)),
            "channel":              r.get("channel", "email"),
            "direction":            r.get("direction", "OUTBOUND"),
            "sent_from":            r.get("sent_from", ""),
            "sent_to":              r.get("sent_to", ""),
            "subject":              r.get("subject", ""),
            "body_preview":         r.get("body_preview", ""),
            "status":               r.get("status", ""),
            "client_response":      r.get("client_response", ""),
            "invoice_amount":       float(r.get("invoice_amount", 0.0)),
            "balance_at_time":      float(r.get("balance_at_time", 0.0)),
            "days_overdue_at_send": int(r.get("days_overdue_at_send", 0)),
            "account_manager":      r.get("account_manager", ""),
        })
    _batch_insert(sb, "communications", rows, "communications")


def migrate_contracts(sb) -> None:
    print("[7/9] Migrating contracts...")
    records = _load_json(DATA_DIR / "contracts.json")
    rows = []
    for r in records:
        rows.append({
            "contract_id":          r["contract_id"],
            "client_id":            r["client_id"],
            "client_name":          r.get("client_name", ""),
            "contract_type":        r.get("contract_type", ""),
            "effective_date":       r["effective_date"],
            "expiry_date":          r["expiry_date"],
            "auto_renew":           r.get("auto_renew", True),
            "renewal_notice_days":  int(r.get("renewal_notice_days", 90)),
            "payment_terms":        r.get("payment_terms", ""),
            "monthly_value":        float(r.get("monthly_value", 0.0)),
            "annual_value":         float(r.get("annual_value", 0.0)),
            "services_in_scope":    r.get("services_in_scope", []),
            "sla_response_time":    r.get("sla_response_time", {}),
            "sla_uptime_guarantee": r.get("sla_uptime_guarantee", ""),
            "late_payment_penalty": r.get("late_payment_penalty", ""),
            "suspension_clause":    r.get("suspension_clause", ""),
            "termination_notice":   r.get("termination_notice", ""),
            "governing_law":        r.get("governing_law", ""),
            "signed_by_client":     r.get("signed_by_client", ""),
            "signed_by_msp":        r.get("signed_by_msp", ""),
        })
    _batch_insert(sb, "contracts", rows, "contracts")


def migrate_rag_chunks(sb) -> None:
    """
    Migrate chunks.json + FAISS embeddings to rag_chunks (pgvector).

    The FAISS index and chunks.json share the same row ordering:
    chunk at position i in chunks.json == vector at row i in faiss.index.
    We use faiss.reconstruct(i) to extract each L2-normalised embedding.
    """
    print("[8/9] Migrating RAG chunks + embeddings...")

    chunks_path = INDEX_DIR / "chunks.json"
    faiss_path  = INDEX_DIR / "faiss.index"

    if not chunks_path.exists():
        print("  [SKIP] data/index/chunks.json not found. Run Phase II first.")
        return
    if not faiss_path.exists():
        print("  [SKIP] data/index/faiss.index not found. Run Phase II first.")
        return

    try:
        import faiss
        import numpy as np
    except ImportError:
        print("  [SKIP] faiss-cpu not installed. Run: pip install faiss-cpu")
        return

    print("  Loading FAISS index...")
    faiss_index = faiss.read_index(str(faiss_path))

    print("  Loading chunks.json...")
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    total = len(chunks)
    if total != faiss_index.ntotal:
        print(
            f"  [WARNING] chunks.json has {total} entries but FAISS has "
            f"{faiss_index.ntotal} vectors. Proceeding with min({total}, "
            f"{faiss_index.ntotal})."
        )
        total = min(total, faiss_index.ntotal)

    # Chunks carry 1536-float embeddings -- each row is ~25 KB of JSON.
    # Use a small batch size (5) to stay well under Supabase's 1 MB body limit.
    CHUNK_BATCH = 5

    print(f"  Extracting {total} embeddings from FAISS and uploading "
          f"(batch size={CHUNK_BATCH})...")
    rows = []
    inserted = 0
    failed = 0

    for i in range(total):
        c = chunks[i]
        vec = faiss_index.reconstruct(i)             # numpy float32 array (1536,)
        # Round to 6 decimal places to cut JSON payload size by ~60 %
        embedding_list = [round(float(v), 6) for v in vec]

        rows.append({
            "chunk_id":       str(c["chunk_id"]),
            "doc_id":         str(c["doc_id"]),
            "chunk_index":    int(c.get("chunk_index", 0)),
            "chunk_strategy": c.get("chunk_strategy", ""),
            "text":           c.get("text", ""),
            "token_count":    int(c.get("token_count", 0)),
            "source":         c.get("source", ""),
            "source_type":    c.get("source_type", ""),
            "title":          c.get("title", ""),
            "url":            c.get("url"),
            "metadata":       c.get("metadata", {}),
            "embedding":      embedding_list,
        })

        if len(rows) == CHUNK_BATCH:
            try:
                # upsert: safe to re-run -- updates existing rows on chunk_id conflict
                sb.table("rag_chunks").upsert(rows, on_conflict="chunk_id").execute()
                inserted += len(rows)
            except Exception as exc:
                failed += len(rows)
                print(f"\n  [ERROR] Batch starting at index {i - CHUNK_BATCH + 1} "
                      f"failed: {exc}")
            rows = []
            print(f"  [rag_chunks] {inserted}/{total} uploaded  "
                  f"({failed} failed)", end="\r")

    if rows:
        try:
            sb.table("rag_chunks").upsert(rows, on_conflict="chunk_id").execute()
            inserted += len(rows)
        except Exception as exc:
            failed += len(rows)
            print(f"\n  [ERROR] Final batch failed: {exc}")

    print(f"  [rag_chunks] {inserted} uploaded, {failed} failed.      ")


def migrate_chat_logs(sb) -> None:
    print("[9/9] Migrating chat logs (JSONL)...")
    if not CHAT_LOG.exists():
        print("  [SKIP] No local chat log found at data/chat_logs/chat_history.jsonl")
        return

    rows = []
    with open(CHAT_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            lat = r.get("latency_ms", {})
            tok = r.get("tokens", {})
            rows.append({
                "created_at":             r.get("timestamp"),
                "session_id":             r.get("session_id", "anonymous"),
                "query":                  r.get("query", ""),
                "answer_length":          int(r.get("answer_length", 0)),
                "provider":               r.get("provider", ""),
                "model":                  r.get("model", ""),
                "blocked":                bool(r.get("blocked", False)),
                "blocked_reason":         r.get("blocked_reason", ""),
                "citation_count":         int(r.get("citation_count", 0)),
                "source_types":           r.get("source_types", []),
                "citations":              r.get("citations", []),
                "latency_retrieval_ms":   float(lat.get("retrieval", 0)),
                "latency_rerank_ms":      float(lat.get("rerank", 0)),
                "latency_generation_ms":  float(lat.get("generation", 0)),
                "latency_total_ms":       float(lat.get("total", 0)),
                "tokens_prompt":          int(tok.get("prompt", 0)),
                "tokens_completion":      int(tok.get("completion", 0)),
                "tokens_total":           int(tok.get("total", 0)),
                "estimated_cost_usd":     float(r.get("estimated_cost_usd", 0.0)),
                "pii_redacted_count":     int(r.get("pii_redacted_count", 0)),
            })

    if not rows:
        print("  [SKIP] Chat log file exists but contains no valid records.")
        return

    _batch_insert(sb, "chat_logs", rows, "chat_logs")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _check_env()

    print("=" * 60)
    print("Red Key Sandbox MSP -- Supabase Data Migration")
    print("=" * 60)
    print(f"  Target: {SUPABASE_URL}")
    print(f"  Batch size: {BATCH_SIZE} rows/call")
    print()

    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--only",
        choices=["clients","invoices","psa","crm","comms","contracts","chunks","logs"],
        default=None,
        help="Re-run a single step only (e.g. --only chunks)",
    )
    args, _ = parser.parse_known_args()

    sb = _client()

    if args.only == "chunks":
        migrate_rag_chunks(sb)
    elif args.only == "logs":
        migrate_chat_logs(sb)
    elif args.only == "clients":
        migrate_clients(sb)
    elif args.only == "invoices":
        migrate_invoices(sb)
    elif args.only == "psa":
        migrate_psa_tickets(sb)
    elif args.only == "crm":
        migrate_crm_profiles(sb)
    elif args.only == "comms":
        migrate_communications(sb)
    elif args.only == "contracts":
        migrate_contracts(sb)
    else:
        migrate_clients(sb)
        migrate_invoices(sb)
        migrate_psa_tickets(sb)
        migrate_crm_profiles(sb)
        migrate_communications(sb)
        migrate_contracts(sb)
        migrate_rag_chunks(sb)
        migrate_chat_logs(sb)

    print()
    print("=" * 60)
    print("Migration complete.")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Add SUPABASE_URL + SUPABASE_SERVICE_KEY to Vercel env vars")
    print("  2. Deploy: vercel --prod")
    print("  3. Re-run this script after any Phase II re-index")


if __name__ == "__main__":
    main()
