-- Migration: Add 'engineer' role to portal_users
-- Allows engineers to have their own portal login accounts

-- Drop the existing check constraint and create a new one with 'engineer' role
ALTER TABLE portal_users DROP CONSTRAINT portal_users_role_check;

ALTER TABLE portal_users ADD CONSTRAINT portal_users_role_check
  CHECK (role IN ('msp', 'client', 'engineer'));

-- Index for fast engineer role lookups (optional, for future bulk queries)
CREATE INDEX IF NOT EXISTS idx_portal_users_role ON portal_users(role);
