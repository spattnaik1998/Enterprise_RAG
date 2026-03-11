#!/usr/bin/env python3
"""
Debug script to diagnose engineer ticket synchronization issues.
Checks:
1. What's in the database (assigned_to values)
2. What the API returns
3. JWT token contents
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv
from supabase import create_client
import json

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_KEY")

if not url or not key:
    print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_KEY not set")
    sys.exit(1)

db = create_client(url, key)

print("\n" + "="*70)
print("ENGINEER TICKET SYNCHRONIZATION DEBUG")
print("="*70)

# 1. Check engineers in database
print("\n[1] ENGINEERS IN DATABASE:")
print("-" * 70)
try:
    res = db.table("engineers").select("id,name,email").execute()
    engineers = res.data or []
    print(f"Total engineers: {len(engineers)}\n")
    for eng in engineers:
        email = eng.get("email", "N/A")
        name = eng.get("name", "N/A")
        username = email.split("@")[0].replace(".", "_").lower() if "@" in email else "N/A"
        print(f"  Name: {name}")
        print(f"  Email: {email}")
        print(f"  Username: {username}")
        print()
except Exception as e:
    print(f"ERROR: {e}")

# 2. Check what's in service_tickets (assigned_to values)
print("\n[2] SERVICE_TICKETS - ASSIGNED_TO VALUES:")
print("-" * 70)
try:
    res = db.table("service_tickets").select("id,ticket_number,assigned_to,title").limit(10).execute()
    tickets = res.data or []
    print(f"Showing first 10 tickets:\n")
    for t in tickets:
        assigned = t.get("assigned_to") or "NULL"
        print(f"  Ticket: {t.get('ticket_number', 'N/A')}")
        print(f"  Title: {t.get('title', 'N/A')[:50]}")
        print(f"  Assigned To: '{assigned}'")
        print(f"  Type: {type(assigned).__name__}")
        print()
except Exception as e:
    print(f"ERROR: {e}")

# 3. Check portal_users for engineers
print("\n[3] ENGINEER ACCOUNTS IN PORTAL_USERS:")
print("-" * 70)
try:
    res = db.table("portal_users").select("username,role,client_id,client_name").eq("role", "engineer").execute()
    users = res.data or []
    print(f"Total engineer accounts: {len(users)}\n")
    for u in users:
        print(f"  Username: {u.get('username', 'N/A')}")
        print(f"  Role: {u.get('role', 'N/A')}")
        print()
except Exception as e:
    print(f"ERROR: {e}")

# 4. Check if Tom Okafor has any assigned tickets
print("\n[4] TICKETS ASSIGNED TO 'tom_okafor':")
print("-" * 70)
try:
    res = db.table("service_tickets").select("id,ticket_number,title,assigned_to").eq("assigned_to", "tom_okafor").execute()
    tickets = res.data or []
    print(f"Tickets with assigned_to='tom_okafor': {len(tickets)}\n")
    for t in tickets:
        print(f"  Ticket #: {t.get('ticket_number', 'N/A')}")
        print(f"  Title: {t.get('title', 'N/A')}")
        print(f"  Assigned To: {t.get('assigned_to', 'N/A')}")
        print()
except Exception as e:
    print(f"ERROR: {e}")

# 5. Check if Tom Okafor's name is in assigned_to
print("\n[5] TICKETS ASSIGNED TO 'Tom Okafor' (FULL NAME):")
print("-" * 70)
try:
    res = db.table("service_tickets").select("id,ticket_number,title,assigned_to").eq("assigned_to", "Tom Okafor").execute()
    tickets = res.data or []
    print(f"Tickets with assigned_to='Tom Okafor': {len(tickets)}\n")
    for t in tickets:
        print(f"  Ticket #: {t.get('ticket_number', 'N/A')}")
        print(f"  Title: {t.get('title', 'N/A')}")
        print(f"  Assigned To: {t.get('assigned_to', 'N/A')}")
        print()
except Exception as e:
    print(f"ERROR: {e}")

# 6. Summary
print("\n[6] DIAGNOSIS SUMMARY:")
print("-" * 70)
try:
    # Count tickets by assigned_to value
    res = db.table("service_tickets").select("assigned_to").execute()
    all_tickets = res.data or []

    assigned_to_values = {}
    for t in all_tickets:
        val = t.get("assigned_to") or "NULL"
        assigned_to_values[val] = assigned_to_values.get(val, 0) + 1

    print("ALL ASSIGNED_TO VALUES IN DATABASE:\n")
    for val, count in sorted(assigned_to_values.items(), key=lambda x: x[1], reverse=True):
        print(f"  '{val}': {count} tickets")

    print("\n" + "="*70)
    if "tom_okafor" in assigned_to_values:
        print("OK: Tickets found with assigned_to='tom_okafor'")
    else:
        print("PROBLEM: No tickets with assigned_to='tom_okafor'")
        if "Tom Okafor" in assigned_to_values:
            print("         But found tickets with assigned_to='Tom Okafor' (full name)")
            print("         FIX: Re-assign tickets using the updated MSP dropdown")

    print("="*70 + "\n")

except Exception as e:
    print(f"ERROR: {e}")
