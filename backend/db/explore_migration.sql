-- EXPLORE & SEARCH FEATURE MIGRATION

-- 1. Enable Extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 2. Update Profiles Table for Trust Metrics
ALTER TABLE profiles 
ADD COLUMN IF NOT EXISTS trust_status VARCHAR(20) DEFAULT 'TRUSTED',
ADD COLUMN IF NOT EXISTS real_percentage INTEGER DEFAULT 100;

-- 2. Update Posts Table for Visibility & Status
ALTER TABLE posts 
ADD COLUMN IF NOT EXISTS visibility VARCHAR(20) DEFAULT 'PUBLIC',
ADD COLUMN IF NOT EXISTS verification_status VARCHAR(20) DEFAULT 'APPROVED';

-- 3. Indexes for Search & Filtering
CREATE INDEX IF NOT EXISTS idx_posts_visibility_status ON posts(visibility, verification_status);
CREATE INDEX IF NOT EXISTS idx_profiles_trust_status ON profiles(trust_status);
CREATE INDEX IF NOT EXISTS idx_posts_caption_gin ON posts USING gin(to_tsvector('english', caption));
CREATE INDEX IF NOT EXISTS idx_profiles_username_trgm ON profiles USING gin(username gin_trgm_ops); -- Requires pg_trgm
-- Note: 'gin_trgm_ops' requires pg_trgm extension. If not enabled, we use standard btree existing on username.

-- 4. Function to Update Profile Trust Stats (Simple Version)
-- This can be called periodically or via trigger. For now, we define it for manual use.
CREATE OR REPLACE FUNCTION update_profile_trust_stats(target_user_id UUID)
RETURNS VOID AS $$
DECLARE
    total INT;
    fake INT;
    pct INT;
    new_status VARCHAR;
BEGIN
    SELECT COUNT(*), COUNT(*) FILTER (WHERE verdict IN ('FAKE', 'REJECTED', 'BLOCK_FAKE'))
    INTO total, fake
    FROM verification_logs
    WHERE user_id = target_user_id;

    IF total > 0 THEN
        pct := 100 - (fake::FLOAT / total::FLOAT * 100)::INT;
    ELSE
        pct := 100;
    END IF;

    IF pct < 70 THEN
        new_status := 'RESTRICTED';
    ELSIF pct < 90 THEN
        new_status := 'AT_RISK';
    ELSE
        new_status := 'TRUSTED';
    END IF;

    UPDATE profiles 
    SET trust_status = new_status, real_percentage = pct
    WHERE id = target_user_id;
END;
$$ LANGUAGE plpgsql;

-- 5. Trigger to Auto-Update Trust Stats on new Verification Log
-- Remove existing trigger if exists
DROP TRIGGER IF EXISTS trigger_update_trust_stats ON verification_logs;

CREATE OR REPLACE FUNCTION trigger_update_trust_stats_func()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM update_profile_trust_stats(NEW.user_id);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_trust_stats
AFTER INSERT ON verification_logs
FOR EACH ROW
EXECUTE FUNCTION trigger_update_trust_stats_func();
