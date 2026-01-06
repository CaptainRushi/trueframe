-- COMMENT SYSTEM LOGIC MIGRATION

-- 1. Add Type and Visibility Columns to Comments
ALTER TABLE comments 
ADD COLUMN IF NOT EXISTS type VARCHAR(20) DEFAULT 'NORMAL', -- NORMAL, CLAIM, QUESTION, CORRECTION
ADD COLUMN IF NOT EXISTS visibility VARCHAR(20) DEFAULT 'VISIBLE', -- VISIBLE, COLLAPSED, HIDDEN, REMOVED
ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS edited_at TIMESTAMPTZ;

-- 2. Indexes for Comment Filtering
CREATE INDEX IF NOT EXISTS idx_comments_post_visibility ON comments(post_id, visibility);
CREATE INDEX IF NOT EXISTS idx_comments_type ON comments(type);

-- 3. Comments Rate Limiting (Optional: Token Bucket or just simple generic)
-- We will handle rate limiting in application logic for MVP simplicity, 
-- or use a simple redis/table approach if needed later.

-- 4. Comment Editing History (Optional separate table, skipping for MVP)
