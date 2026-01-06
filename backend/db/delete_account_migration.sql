-- SQL Migration for Account Deletion
-- Run this in Supabase SQL Editor

-- 1. Add is_deleted flag to profiles
ALTER TABLE profiles 
ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN DEFAULT FALSE;

-- 2. Allow posts to be anonymized (orphan from user)
ALTER TABLE posts 
ALTER COLUMN user_id DROP NOT NULL;

-- 3. Drop Foreign Key on profiles.id -> auth.users.id
-- This allows us to DELETE the auth user (preventing login) while KEEPING the profile data (for verification logs/history)
ALTER TABLE profiles 
DROP CONSTRAINT IF EXISTS profiles_id_fkey;

COMMENT ON COLUMN profiles.username IS 'Username, replaced with deleted_user_<uuid> upon deletion';
