-- verification_logs table update
-- Ensure we have the granular columns to track specific failure points
-- and a single authoritative final_verdict.

ALTER TABLE verification_logs 
ADD COLUMN IF NOT EXISTS deepfake_verdict VARCHAR(50), 
ADD COLUMN IF NOT EXISTS fake_news_verdict VARCHAR(50),
ADD COLUMN IF NOT EXISTS final_verdict VARCHAR(50);

-- Migrate existing data (Best Effort)
-- If verdict was 'REAL', assume deepfake=APPROVED, fake_news=APPROVED, final=REAL
UPDATE verification_logs 
SET 
  final_verdict = verdict,
  deepfake_verdict = CASE WHEN verdict = 'REAL' THEN 'APPROVED' ELSE 'REJECTED' END,
  fake_news_verdict = CASE WHEN verdict = 'REAL' THEN 'APPROVED' ELSE 'SKIPPED' END
WHERE final_verdict IS NULL;

-- Index for fast counting
CREATE INDEX IF NOT EXISTS idx_logs_user_final ON verification_logs(user_id, final_verdict);
