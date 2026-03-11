# Engineer Portal & MSP Client Management — Deployment Guide

## Overview

Two new portals have been implemented:

1. **Engineer Portal** — Engineers can login, view only their assigned tickets, and update ticket status/notes
2. **Manage Clients Tab** — MSP admins can view all client credentials with copy-to-clipboard functionality

## Pre-Deployment Checklist

All code is committed and ready. The application is fully functional pending one manual database migration step.

## Deployment Steps

### Step 1: Execute Database Migration ⚠️ REQUIRED

The engineer role must be added to the `portal_users` table before engineer accounts can be created.

**Option A: Via Supabase Dashboard (Recommended)**

1. Go to https://supabase.com/dashboard
2. Select your Red Key Sandbox project
3. Click **SQL Editor** in the left sidebar
4. Click **New Query**
5. Paste this SQL:

```sql
-- Migration: Add 'engineer' role to portal_users
-- Allows engineers to have their own portal login accounts

ALTER TABLE portal_users DROP CONSTRAINT portal_users_role_check;

ALTER TABLE portal_users ADD CONSTRAINT portal_users_role_check
  CHECK (role IN ('msp', 'client', 'engineer'));

CREATE INDEX IF NOT EXISTS idx_portal_users_role ON portal_users(role);
```

6. Click **Run**

**Option B: Via Supabase CLI**

```bash
npx supabase db push
```

**Verify migration was successful:**

Go to the SQL Editor and run:
```sql
SELECT constraint_name, constraint_definition FROM information_schema.check_constraints
WHERE table_name = 'portal_users' AND constraint_name = 'portal_users_role_check';
```

Expected output: Should show the role check constraint with `('msp', 'client', 'engineer')`.

---

### Step 2: Create Engineer Login Accounts

After the migration succeeds, create engineer accounts with generated passwords:

```bash
python scripts/setup_engineer_credentials.py
```

This script will:
- Extract all unique engineers from PSA tickets
- Create login accounts for each engineer
- Generate secure random passwords
- Display credentials in a table

**Output Example:**
```
Created 8 engineer accounts:
┌─────────────────┬────────────────────┬──────────────────────┐
│ Engineer        │ Username           │ Generated Password   │
├─────────────────┼────────────────────┼──────────────────────┤
│ Alex Rivera     │ alex_rivera        │ m$K2pQx!9Dv#L        │
│ Jamie Chen      │ jamie_chen         │ 8Fq@nR$yJ2k5X        │
│ ...             │ ...                │ ...                  │
└─────────────────┴────────────────────┴──────────────────────┘

All accounts created. Store these credentials in your password manager.
Engineers can now login at /engineer
```

---

### Step 3: Start the Application

```bash
uvicorn app.server:app --reload --port 8000
```

Navigate to http://localhost:8000

---

## Portal Access

### Client Portal
- **URL**: http://localhost:8000
- **Role**: Client users submit tickets
- **Who**: Clients (existing credentials unchanged)

### Engineer Portal
- **URL**: http://localhost:8000/engineer
- **Role**: Engineers view and update their assigned tickets
- **Who**: Engineers (new accounts created in Step 2)
- **Features**:
  - Dashboard with ticket status breakdown
  - Ticket list filterable by status
  - Ticket detail panel with status update and notes
  - Real-time stats updates across all engineer instances

### MSP Admin Portal
- **URL**: http://localhost:8000/msp
- **Role**: MSP administrators manage all tickets and client credentials
- **Who**: MSP staff (existing accounts unchanged)
- **New Features**:
  - **Engineers Tab**: View all engineers with ticket stats and SLA metrics
  - **Manage Clients Tab**: View all client credentials with copy buttons
    - Search clients by name or username
    - Copy username to clipboard
    - Copy password to clipboard
    - Direct links to client portal accounts

---

## Testing Workflow

### 1. Test Client → Ticket Creation
```bash
# Open client portal
# Login: clt-001 / whvro1rSh$AH
# Client name: Lakewood Medical Group
# Create a new ticket
```

### 2. Test MSP → Assign to Engineer
```bash
# Open MSP portal
# Navigate to Service Tickets
# Select a ticket
# Assign to an engineer from the dropdown
# Update status to "in_progress"
```

### 3. Test Engineer Portal
```bash
# Open engineer portal
# Login with engineer credentials from Step 2
# Should see only assigned tickets
# Update status to "waiting_client" or "resolved"
# Verify stats update in real-time
```

### 4. Test MSP → View Clients
```bash
# Open MSP portal
# Click "Manage Clients" tab
# Click copy button on a username
# Paste to verify it copied correctly
# Click copy button on a password
# Verify password copied (appears masked in UI)
```

---

## Database Schema

### portal_users table extensions

```sql
role IN ('msp', 'client', 'engineer')  -- Added 'engineer'
```

### New Functions in app/portal_db.py

| Function | Purpose |
|----------|---------|
| `get_engineer_tickets(engineer_name)` | List all tickets for engineer |
| `get_engineer_ticket_by_id(engineer_name, ticket_id)` | Get ticket with ownership check |
| `update_engineer_ticket(engineer_name, ticket_id, updates)` | Update with auth verification |
| `get_engineer_stats(engineer_name)` | Get ticket status breakdown |
| `get_all_client_credentials()` | Return client credentials with passwords |

---

## API Endpoints

### Engineer Endpoints (require `require_engineer` auth)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/engineer` | Serve engineer portal HTML |
| GET | `/api/engineer/tickets` | List engineer's assigned tickets |
| GET | `/api/engineer/tickets/{id}` | Get single ticket (ownership check) |
| PATCH | `/api/engineer/tickets/{id}` | Update ticket (status, notes) |
| GET | `/api/engineer/stats` | Get ticket status breakdown |

### MSP Endpoints (require `require_msp` auth)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/msp/clients/credentials` | All client credentials (new) |
| GET | `/api/msp/engineers` | All engineer profiles (existing) |
| GET | `/api/msp/ticket-stats` | Overall ticket statistics (existing) |

---

## Authorization Model

### Engineer Portal Access Control

Engineers can **only**:
- View tickets assigned to them (`assigned_to == engineer_name`)
- Update status and notes on their own tickets
- See their own ticket statistics

Database-level enforcement in `update_engineer_ticket()`:
```python
ticket = get_engineer_ticket_by_id(engineer_name, ticket_id)
if ticket is None:  # Engineer does not own this ticket
    return None  # Unauthorized
```

### MSP Portal Access Control

MSP admins can:
- View all tickets across all clients and engineers
- Assign tickets to engineers
- Update any ticket
- View all client credentials
- View all engineer profiles

---

## File Structure

```
app/
├── server.py                          (6 engineer endpoints + 1 MSP endpoint)
├── auth.py                            (+ require_engineer guard)
├── portal_db.py                       (+ 6 engineer functions)
└── static/
    ├── engineer_portal.html           (NEW - 1300+ lines)
    └── msp_portal.html                (+ Manage Clients tab)

scripts/
├── setup_engineer_credentials.py      (NEW - Create engineer accounts)
├── execute_migration.py               (NEW - Migration execution helper)
└── run_migration.py                   (NEW - Migration display helper)

supabase/migrations/
└── 20260311_add_engineer_role.sql     (NEW - Schema migration)
```

---

## Troubleshooting

### "violates check constraint portal_users_role_check"

**Cause**: Migration SQL hasn't been executed
**Fix**: Run the SQL in Supabase dashboard (Step 1)

### Engineer Portal shows "Unauthorized" or redirects to login

**Cause**: Engineer account not created yet
**Fix**: Run `python scripts/setup_engineer_credentials.py` (Step 2)

### Manage Clients tab shows "Error loading clients"

**Cause**: API endpoint not authenticated
**Fix**: Verify MSP is logged in with valid JWT token

### Copy button doesn't copy to clipboard

**Cause**: Browser security policy (HTTPS required in production)
**Fix**: Works fine in development (http://localhost), test in HTTP

---

## Next Steps (Post-Deployment)

1. ✓ All code implemented and committed
2. ⏳ User runs SQL migration in Supabase (Step 1)
3. ⏳ User creates engineer accounts (Step 2)
4. ⏳ User tests all three portals
5. Deploy to production (Vercel or cloud provider)

---

## Git Commit

```
Commit: 5177db7
feat: Add engineer portal and manage clients tab for MSP
- Engineer Portal with dashboard and ticket management
- Manage Clients tab with credentials distribution
- 6 new engineer API endpoints + 1 MSP endpoint
- Database functions for authorization
- Schema migration for engineer role
```

---

## Questions?

Refer to:
- **CLAUDE.md** — Architecture and command reference
- **app/portal_db.py** — Database function documentation
- **app/server.py** — API endpoint definitions
- **app/static/engineer_portal.html** — Frontend implementation
- **app/static/msp_portal.html** — MSP dashboard with new tabs
