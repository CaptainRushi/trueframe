-- ==========================================
-- UNIFIED SCHEMA SCRIPT (V5)
-- ==========================================
-- This file contains the base schema and all advanced features
-- combined into a single executable script for Supabase SQL Editor.
-- ==========================================

-- Enable UUID Extension and pg_trgm for search
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ==========================================
-- 1. BASE TABLES
-- ==========================================

-- Public Profiles Table
CREATE TABLE IF NOT EXISTS profiles (
    id UUID PRIMARY KEY, -- Removed references auth.users(id) for deleted accounts logic
    username VARCHAR(50) UNIQUE NOT NULL, -- Replaced with deleted_user_<uuid> upon deletion
    display_name VARCHAR(100),
    avatar_url TEXT,
    bio TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    followers_count INTEGER DEFAULT 0,
    following_count INTEGER DEFAULT 0,
    trust_status VARCHAR(20) DEFAULT 'NEW_USER',
    real_percentage INTEGER DEFAULT 100,
    trust_score INTEGER DEFAULT 50,
    trust_score_updated_at TIMESTAMP WITH TIME ZONE,
    fake_percentage INTEGER DEFAULT 0,
    total_attempts INTEGER DEFAULT 0,
    real_count INTEGER DEFAULT 0,
    fake_count INTEGER DEFAULT 0,
    is_deleted BOOLEAN DEFAULT FALSE,
    is_verified_creator BOOLEAN DEFAULT FALSE,
    verified_creator_at TIMESTAMP WITH TIME ZONE,
    is_community_verifier BOOLEAN DEFAULT FALSE,
    identity_verified BOOLEAN DEFAULT FALSE
);

COMMENT ON COLUMN profiles.fake_percentage IS 'Cached percentage of rejected uploads';
COMMENT ON COLUMN profiles.total_attempts IS 'Cached total number of verification attempts';
COMMENT ON COLUMN profiles.real_count IS 'Cached count of approved real uploads';
COMMENT ON COLUMN profiles.fake_count IS 'Cached count of rejected fake uploads';
COMMENT ON COLUMN profiles.username IS 'Username, replaced with deleted_user_<uuid> upon deletion';

-- Verification Logs
CREATE TABLE IF NOT EXISTS verification_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES profiles(id),
    media_hash VARCHAR(64) NOT NULL,
    verification_type VARCHAR(20) DEFAULT 'MEDIA' CHECK (verification_type IN ('MEDIA', 'CONTEXT')),
    verdict VARCHAR(50) NOT NULL,
    score DECIMAL(5, 4) NOT NULL,
    reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    authenticity_label VARCHAR(50) DEFAULT 'UNKNOWN',
    score_breakdown JSONB,
    deepfake_verdict VARCHAR(50),
    fake_news_verdict VARCHAR(50),
    final_verdict VARCHAR(50),
    model_name VARCHAR(50) DEFAULT 'efficientnet-b0',
    model_version VARCHAR(20) DEFAULT '1.0',
    model_score FLOAT,
    final_score FLOAT,
    media_type VARCHAR(10),
    upload_source VARCHAR(20) DEFAULT 'GALLERY',
    device_metadata JSONB
);

-- Posts Table
CREATE TABLE IF NOT EXISTS posts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES profiles(id), -- Nullable for deleted accounts
    media_url TEXT NOT NULL,
    media_type VARCHAR(10) CHECK (media_type IN ('image', 'video')) NOT NULL,
    caption TEXT,
    media_hash_check VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    verification_log_id UUID REFERENCES verification_logs(id),
    like_count INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    share_count INTEGER DEFAULT 0,
    visibility VARCHAR(20) DEFAULT 'PUBLIC',
    verification_status VARCHAR(20) DEFAULT 'APPROVED',
    authenticity_label VARCHAR(50) DEFAULT 'VERIFIED_REAL',
    upload_source VARCHAR(20) DEFAULT 'GALLERY',
    content_hash_proof VARCHAR(128)
);

-- ==========================================
-- 2. SOCIAL FEATURES & ADVANCED TABLES
-- ==========================================

-- Likes
CREATE TABLE IF NOT EXISTS likes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(post_id, user_id)
);

-- Comments
CREATE TABLE IF NOT EXISTS comments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    type VARCHAR(20) DEFAULT 'NORMAL', 
    visibility VARCHAR(20) DEFAULT 'VISIBLE', 
    is_deleted BOOLEAN DEFAULT FALSE,
    edited_at TIMESTAMPTZ,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Shares
CREATE TABLE IF NOT EXISTS shares (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Follows
CREATE TABLE IF NOT EXISTS follows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    follower_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    following_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(follower_id, following_id),
    CHECK (follower_id <> following_id)
);

-- Trust Score History
CREATE TABLE IF NOT EXISTS trust_score_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    trust_score INTEGER NOT NULL,
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

-- Community Flags
CREATE TABLE IF NOT EXISTS community_flags (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    flagger_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    flag_type VARCHAR(30) NOT NULL CHECK (flag_type IN ('MISINFORMATION', 'MANIPULATED', 'OUT_OF_CONTEXT', 'SPAM', 'OTHER')),
    reason TEXT,
    source_url TEXT,
    status VARCHAR(20) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'CONFIRMED', 'DISMISSED')),
    reviewed_by UUID REFERENCES profiles(id),
    reviewed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(post_id, flagger_id)
);

-- Notifications
CREATE TABLE IF NOT EXISTS notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    type VARCHAR(30) NOT NULL CHECK (type IN (
        'LIKE', 'COMMENT', 'FOLLOW', 'VERIFICATION_PASSED', 'VERIFICATION_FAILED',
        'TRUST_CHANGE', 'CREATOR_BADGE', 'FLAG_RESULT', 'DEEPFAKE_ALERT', 'SYSTEM'
    )),
    title TEXT NOT NULL,
    message TEXT,
    related_post_id UUID REFERENCES posts(id) ON DELETE SET NULL,
    related_user_id UUID REFERENCES profiles(id) ON DELETE SET NULL,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Content Proofs
CREATE TABLE IF NOT EXISTS content_proofs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE UNIQUE,
    media_hash VARCHAR(64) NOT NULL,
    metadata_hash VARCHAR(64) NOT NULL,
    proof_hash VARCHAR(128) NOT NULL,
    previous_proof_hash VARCHAR(128),
    proof_chain_index INTEGER NOT NULL,
    verified_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Deepfake Alerts
CREATE TABLE IF NOT EXISTS deepfake_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    description TEXT,
    media_hash VARCHAR(64),
    detection_count INTEGER DEFAULT 1,
    severity VARCHAR(10) DEFAULT 'MEDIUM' CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================
-- 3. INDEXES
-- ==========================================
CREATE INDEX IF NOT EXISTS idx_profiles_username ON profiles(username);
CREATE INDEX IF NOT EXISTS idx_profiles_trust_status ON profiles(trust_status);
CREATE INDEX IF NOT EXISTS idx_profiles_trust_score ON profiles(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_profiles_username_trgm ON profiles USING gin(username gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_verification_logs_user_id ON verification_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_verification_logs_media_hash ON verification_logs(media_hash);

CREATE INDEX IF NOT EXISTS idx_posts_user_id ON posts(user_id);
CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_posts_visibility_status ON posts(visibility, verification_status);
CREATE INDEX IF NOT EXISTS idx_posts_caption_gin ON posts USING gin(to_tsvector('english', caption));

CREATE INDEX IF NOT EXISTS idx_likes_post_id ON likes(post_id);
CREATE INDEX IF NOT EXISTS idx_likes_user_id ON likes(user_id);

CREATE INDEX IF NOT EXISTS idx_comments_post_id ON comments(post_id);
CREATE INDEX IF NOT EXISTS idx_comments_post_visibility ON comments(post_id, visibility);
CREATE INDEX IF NOT EXISTS idx_comments_type ON comments(type);

CREATE INDEX IF NOT EXISTS idx_shares_post_id ON shares(post_id);

CREATE INDEX IF NOT EXISTS idx_follows_follower_id ON follows(follower_id);
CREATE INDEX IF NOT EXISTS idx_follows_following_id ON follows(following_id);

CREATE INDEX IF NOT EXISTS idx_trust_history_user_date ON trust_score_history(user_id, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_community_flags_post ON community_flags(post_id);
CREATE INDEX IF NOT EXISTS idx_community_flags_status ON community_flags(status);

CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_notifications_unread ON notifications(user_id, is_read) WHERE is_read = FALSE;

CREATE INDEX IF NOT EXISTS idx_content_proofs_post ON content_proofs(post_id);
CREATE INDEX IF NOT EXISTS idx_content_proofs_chain ON content_proofs(proof_chain_index);

-- ==========================================
-- 4. RPC FUNCTIONS & TRIGGERS
-- ==========================================

-- Counters
CREATE OR REPLACE FUNCTION increment_like_count(post_id UUID)
RETURNS void AS $$
BEGIN
  UPDATE posts SET like_count = like_count + 1 WHERE id = post_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION decrement_like_count(post_id UUID)
RETURNS void AS $$
BEGIN
  UPDATE posts SET like_count = GREATEST(like_count - 1, 0) WHERE id = post_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION increment_comment_count(post_id UUID)
RETURNS void AS $$
BEGIN
  UPDATE posts SET comment_count = comment_count + 1 WHERE id = post_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION increment_share_count(post_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE posts SET share_count = share_count + 1 WHERE id = post_id;
END;
$$ LANGUAGE plpgsql;

-- Follow Counts
CREATE OR REPLACE FUNCTION update_follow_counts()
RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'INSERT') THEN
        UPDATE profiles SET following_count = following_count + 1 WHERE id = NEW.follower_id;
        UPDATE profiles SET followers_count = followers_count + 1 WHERE id = NEW.following_id;
        RETURN NEW;
    ELSIF (TG_OP = 'DELETE') THEN
        UPDATE profiles SET following_count = GREATEST(following_count - 1, 0) WHERE id = OLD.follower_id;
        UPDATE profiles SET followers_count = GREATEST(followers_count - 1, 0) WHERE id = OLD.following_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_follow_counts ON follows;
CREATE TRIGGER trigger_update_follow_counts
AFTER INSERT OR DELETE ON follows
FOR EACH ROW
EXECUTE FUNCTION update_follow_counts();

-- Trust Stats
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
