-- Migration to add granular trust metrics to profiles
ALTER TABLE profiles 
ADD COLUMN IF NOT EXISTS fake_percentage INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS total_attempts INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS real_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS fake_count INTEGER DEFAULT 0;

-- Comment for documentation
COMMENT ON COLUMN profiles.fake_percentage IS 'Cached percentage of rejected uploads';
COMMENT ON COLUMN profiles.total_attempts IS 'Cached total number of verification attempts';
COMMENT ON COLUMN profiles.real_count IS 'Cached count of approved real uploads';
COMMENT ON COLUMN profiles.fake_count IS 'Cached count of rejected fake uploads';
