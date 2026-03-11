-- Migration: Create engineers table for Red Key Sandbox MSP portal
-- Run this once in the Supabase SQL Editor (or via psql)

CREATE TABLE IF NOT EXISTS engineers (
    id                    BIGSERIAL PRIMARY KEY,
    name                  TEXT NOT NULL UNIQUE,
    email                 TEXT NOT NULL UNIQUE,
    total_tickets         INTEGER DEFAULT 0,
    resolved_tickets      INTEGER DEFAULT 0,
    avg_hours_per_ticket  NUMERIC(6,2) DEFAULT 0,
    sla_rate              NUMERIC(5,1) DEFAULT 0,
    primary_specialty     TEXT DEFAULT 'other',
    ticket_type_breakdown JSONB DEFAULT '{}',
    created_at            TIMESTAMPTZ DEFAULT now(),
    updated_at            TIMESTAMPTZ DEFAULT now()
);

-- Enable RLS (service role bypasses, anon gets read-only)
ALTER TABLE engineers ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow service role full access"
    ON engineers FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow anon read"
    ON engineers FOR SELECT
    USING (true);
