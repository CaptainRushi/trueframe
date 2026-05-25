-- ==========================================
-- MIGRATION: Secondary Review System
-- ==========================================
-- Adds community reporting threshold → secondary AI review pipeline

-- Secondary Reviews table (one-to-one with posts)
CREATE TABLE IF NOT EXISTS secondary_reviews (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    triggered_by VARCHAR(20) NOT NULL DEFAULT 'COMMUNITY_FLAGS'
        CHECK (triggered_by IN ('COMMUNITY_FLAGS', 'MANUAL', 'SYSTEM')),
    trigger_flag_count INTEGER NOT NULL DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING'
        CHECK (status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED')),

    -- Secondary AI scores (frequency-domain & GAN-artifact focus)
    secondary_score FLOAT,
    frequency_score FLOAT,
    gan_artifact_score FLOAT,
    noise_consistency_score FLOAT,
    edge_coherence_score FLOAT,
    patch_variance_score FLOAT,
    score_breakdown JSONB,
    signals TEXT[],
    swin_score FLOAT,
    swin_risk_tier VARCHAR(20)
        CHECK (swin_risk_tier IN ('LOW', 'ELEVATED', 'HIGH', 'UNAVAILABLE')),

    -- Automated decision
    decision VARCHAR(20)
        CHECK (decision IN ('RESTORE', 'MANUAL_REVIEW', 'REMOVE')),
    decision_reason TEXT,

    -- Manual review fields (for scores in 0.60-0.80 range)
    manual_reviewer_id UUID REFERENCES profiles(id),
    manual_decision VARCHAR(20)
        CHECK (manual_decision IN ('RESTORE', 'REMOVE')),
    manual_review_reason TEXT,
    manual_reviewed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,

    UNIQUE(post_id)
);

-- Add flag weight tracking for abuse prevention
ALTER TABLE community_flags ADD COLUMN IF NOT EXISTS flag_weight FLOAT DEFAULT 1.0;
ALTER TABLE community_flags ADD COLUMN IF NOT EXISTS triggered_review BOOLEAN DEFAULT FALSE;

-- Add Swin-L tertiary review fields for manual review risk ranking
ALTER TABLE secondary_reviews ADD COLUMN IF NOT EXISTS swin_score FLOAT;
ALTER TABLE secondary_reviews ADD COLUMN IF NOT EXISTS swin_risk_tier VARCHAR(20)
    CHECK (swin_risk_tier IN ('LOW', 'ELEVATED', 'HIGH', 'UNAVAILABLE'));

-- Indexes for secondary reviews
CREATE INDEX IF NOT EXISTS idx_secondary_reviews_post ON secondary_reviews(post_id);
CREATE INDEX IF NOT EXISTS idx_secondary_reviews_status ON secondary_reviews(status);
CREATE INDEX IF NOT EXISTS idx_secondary_reviews_decision ON secondary_reviews(decision);

-- Update notification type constraint to support new types
ALTER TABLE notifications DROP CONSTRAINT IF EXISTS notifications_type_check;
ALTER TABLE notifications ADD CONSTRAINT notifications_type_check
    CHECK (type IN (
        'LIKE', 'COMMENT', 'FOLLOW', 'VERIFICATION_PASSED', 'VERIFICATION_FAILED',
        'TRUST_CHANGE', 'CREATOR_BADGE', 'FLAG_RESULT', 'DEEPFAKE_ALERT', 'SYSTEM',
        'SECONDARY_REVIEW_RESULT', 'POST_REMOVED', 'TRUST_PENALTY'
    ));
