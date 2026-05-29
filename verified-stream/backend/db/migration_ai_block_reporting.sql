-- ====================================================================
-- TRUEFRAME MIGRATION: AI-Generated Block + Automated Report Pipeline
-- ====================================================================
-- Paste this into your Supabase SQL Editor to apply.
-- Run after schema_complete.sql (or after an existing deployment).
-- ====================================================================

-- 1. Extend community_flags flag_type to include AI_GENERATED and CARTOON
--    Users can now explicitly report posts they believe are AI-generated
--    or illustrated/cartoon content.
ALTER TABLE community_flags
  DROP CONSTRAINT IF EXISTS community_flags_flag_type_check;

ALTER TABLE community_flags
  ADD CONSTRAINT community_flags_flag_type_check
  CHECK (flag_type IN (
    'MISINFORMATION',
    'MANIPULATED',
    'OUT_OF_CONTEXT',
    'SPAM',
    'AI_GENERATED',
    'CARTOON',
    'OTHER'
  ));

-- 2. Add Model 2 result columns to secondary_reviews table.
--    When Model 1 score is borderline (0.60-0.80), Model 2 (main.py
--    HuggingFace+signal) is automatically run. Its results are stored here.
ALTER TABLE secondary_reviews
  ADD COLUMN IF NOT EXISTS model2_score      FLOAT,
  ADD COLUMN IF NOT EXISTS model2_signals    TEXT[],
  ADD COLUMN IF NOT EXISTS model2_decision   VARCHAR(20)
    CHECK (model2_decision IN ('RESTORE', 'REMOVE'));

-- 3. Expand secondary_reviews.decision to include the transient 'RUN_MODEL_2'
--    internal state. In practice this is written then immediately overwritten
--    when Model 2 completes, but allow it for observability/debugging.
ALTER TABLE secondary_reviews
  DROP CONSTRAINT IF EXISTS secondary_reviews_decision_check;

ALTER TABLE secondary_reviews
  ADD CONSTRAINT secondary_reviews_decision_check
  CHECK (decision IN ('RESTORE', 'MANUAL_REVIEW', 'REMOVE', 'RUN_MODEL_2'));

-- 4. Index on model2_decision for quick aggregation queries
CREATE INDEX IF NOT EXISTS idx_secondary_reviews_model2_decision
  ON secondary_reviews(model2_decision);

-- 5. Add new notification type for content-type blocking events
--    (AI-generated and cartoon blocks now emit VERIFICATION_FAILED
--    notifications — the existing check already includes this type, 
--    no change needed unless you want a distinct type).
-- COMMENT: No notification table changes required; VERIFICATION_FAILED is reused.

COMMENT ON COLUMN secondary_reviews.model2_score IS
  'Final score from Model 2 (main.py HuggingFace+signal). Populated when Model 1 score is borderline (0.60-0.80).';

COMMENT ON COLUMN secondary_reviews.model2_decision IS
  'Final decision produced by Model 2: RESTORE (authentic) or REMOVE (fake/unsafe). NULL if Model 1 was decisive.';
