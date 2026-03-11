-- ==========================================
-- IDENTITY VERIFICATION MIGRATION
-- ==========================================
-- Adds selfie-based identity verification system

-- Add verification columns to profiles (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'profiles' AND column_name = 'identity_verified_at') THEN
        ALTER TABLE profiles ADD COLUMN identity_verified_at TIMESTAMP WITH TIME ZONE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'profiles' AND column_name = 'identity_verification_method') THEN
        ALTER TABLE profiles ADD COLUMN identity_verification_method VARCHAR(30);
    END IF;
END $$;

-- Identity Verifications Table
CREATE TABLE IF NOT EXISTS identity_verifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'PENDING_REVIEW', 'FAILED')),
    verdict VARCHAR(20) CHECK (verdict IN ('VERIFIED', 'REVIEW', 'REJECTED', 'ERROR')),
    final_score FLOAT,
    liveness_score FLOAT,
    spoof_score FLOAT,
    ai_face_score FLOAT,
    deepfake_score FLOAT,
    signals TEXT[],
    reason TEXT,
    processing_ms INTEGER,
    device_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP WITH TIME ZONE,
    reviewed_by UUID REFERENCES profiles(id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_identity_verifications_user ON identity_verifications(user_id);
CREATE INDEX IF NOT EXISTS idx_identity_verifications_status ON identity_verifications(status);
CREATE INDEX IF NOT EXISTS idx_identity_verifications_created ON identity_verifications(created_at DESC);

-- Add VERIFICATION_RESULT to notification types if not already present
-- (The existing CHECK constraint allows 'SYSTEM' type which we use instead)
