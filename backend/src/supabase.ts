import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
dotenv.config();

const supabaseUrl = (process.env.SUPABASE_URL || '').trim();
const supabaseKey = (process.env.SUPABASE_SERVICE_ROLE_KEY || '').trim();

if (!supabaseUrl || !supabaseKey) {
    console.error('CRITICAL: Supabase environment variables missing in backend. Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY');
}

// Will intentionally throw if URL/Key are empty during client usage so we can track it down
const supabase = (supabaseUrl && supabaseKey) 
    ? createClient(supabaseUrl, supabaseKey) 
    : null as any;

export { supabase };
