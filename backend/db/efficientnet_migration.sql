-- Migration to add EfficientNet specific metrics
ALTER TABLE verification_logs 
ADD COLUMN IF NOT EXISTS model_name VARCHAR(50) DEFAULT 'efficientnet-b0',
ADD COLUMN IF NOT EXISTS model_version VARCHAR(20) DEFAULT '1.0',
ADD COLUMN IF NOT EXISTS model_score FLOAT,
ADD COLUMN IF NOT EXISTS final_score FLOAT;

-- Update existing logs to have legacy values if needed
UPDATE verification_logs 
SET final_score = score, model_score = score
WHERE final_score IS NULL;
