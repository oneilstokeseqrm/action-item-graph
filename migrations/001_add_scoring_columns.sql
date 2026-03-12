-- Migration: Add scoring and quality columns to action_items table
-- Phase 4: Scoring & Surfacing
-- Date: 2026-03-12
--
-- These columns promote scoring from the flexible `attributes` JSONB/`ai_suggestion_details`
-- to first-class columns, enabling indexed queries and priority-based retrieval.

ALTER TABLE action_items ADD COLUMN IF NOT EXISTS commitment_strength TEXT;
ALTER TABLE action_items ADD COLUMN IF NOT EXISTS score_impact INTEGER;
ALTER TABLE action_items ADD COLUMN IF NOT EXISTS score_urgency INTEGER;
ALTER TABLE action_items ADD COLUMN IF NOT EXISTS score_specificity INTEGER;
ALTER TABLE action_items ADD COLUMN IF NOT EXISTS score_effort INTEGER;
ALTER TABLE action_items ADD COLUMN IF NOT EXISTS priority_score FLOAT;
ALTER TABLE action_items ADD COLUMN IF NOT EXISTS definition_of_done TEXT;

-- Index for priority-based retrieval: fast lookup of top-priority open items per account
CREATE INDEX IF NOT EXISTS idx_action_items_priority
ON action_items (tenant_id, account_id, priority_score DESC NULLS LAST)
WHERE status = 'pending';
