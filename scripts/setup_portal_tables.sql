-- =============================================================================
-- TechVault Client Portal -- Supabase Table Setup
-- =============================================================================
-- Run this in the Supabase SQL Editor (Dashboard > SQL Editor > New query)
-- before running scripts/setup_portal.py
-- =============================================================================

-- ---------------------------------------------------------------------------
-- portal_users: one row per login (1 MSP admin + 50 client accounts)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS portal_users (
    id           uuid        DEFAULT gen_random_uuid() PRIMARY KEY,
    username     text        UNIQUE NOT NULL,
    password_hash text       NOT NULL,
    role         text        NOT NULL CHECK (role IN ('msp', 'client')),
    client_id    text,          -- NULL for msp; 'CLT-001' etc. for clients
    client_name  text,          -- NULL for msp; display name for clients
    created_at   timestamptz DEFAULT now(),
    last_login   timestamptz
);

-- Index for fast username lookups on login
CREATE INDEX IF NOT EXISTS idx_portal_users_username ON portal_users(username);
CREATE INDEX IF NOT EXISTS idx_portal_users_client_id ON portal_users(client_id);

-- ---------------------------------------------------------------------------
-- service_tickets: tickets created through the client portal
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS service_tickets (
    id             uuid        DEFAULT gen_random_uuid() PRIMARY KEY,
    ticket_number  text        UNIQUE NOT NULL,
    client_id      text        NOT NULL,
    client_name    text        NOT NULL,
    title          text        NOT NULL,
    description    text        NOT NULL,
    priority       text        NOT NULL DEFAULT 'medium'
                               CHECK (priority IN ('low', 'medium', 'high', 'critical')),
    category       text        NOT NULL DEFAULT 'other'
                               CHECK (category IN (
                                   'network', 'hardware', 'software',
                                   'security', 'email', 'cloud', 'backup', 'other'
                               )),
    status         text        NOT NULL DEFAULT 'open'
                               CHECK (status IN (
                                   'open', 'in_progress', 'waiting_client',
                                   'resolved', 'closed'
                               )),
    assigned_to    text,           -- engineer name assigned by MSP
    engineer_notes text,           -- internal notes from engineer
    created_at     timestamptz DEFAULT now(),
    updated_at     timestamptz DEFAULT now(),
    resolved_at    timestamptz
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_service_tickets_client_id  ON service_tickets(client_id);
CREATE INDEX IF NOT EXISTS idx_service_tickets_status     ON service_tickets(status);
CREATE INDEX IF NOT EXISTS idx_service_tickets_created_at ON service_tickets(created_at DESC);

-- ---------------------------------------------------------------------------
-- Auto-update updated_at on every row change
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_service_tickets_updated_at ON service_tickets;
CREATE TRIGGER trg_service_tickets_updated_at
    BEFORE UPDATE ON service_tickets
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ---------------------------------------------------------------------------
-- Verification
-- ---------------------------------------------------------------------------
SELECT 'portal_users table ready'     AS status WHERE EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'portal_users');
SELECT 'service_tickets table ready'  AS status WHERE EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'service_tickets');
