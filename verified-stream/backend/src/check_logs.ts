import { supabase } from './supabase.js';

async function checkLogs() {
    if (!supabase) {
        console.error("Supabase client is null!");
        return;
    }
    const { data, error } = await supabase
        .from('verification_logs')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(5);

    if (error) {
        console.error("Error fetching logs:", error);
    } else {
        console.log("Recent logs:", JSON.stringify(data, null, 2));
    }
}

checkLogs();
