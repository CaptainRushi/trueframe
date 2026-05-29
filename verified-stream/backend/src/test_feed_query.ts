import { supabase } from './supabase.js';

async function testQuery() {
    console.log("Starting profiles query test...");
    const { data, error } = await supabase
        .from('profiles')
        .select('*')
        .limit(5);

    if (error) {
        console.error("Profiles Query Error:", error);
    } else {
        console.log("Profiles Query Success! Number of profiles:", data?.length);
        console.log("Sample profile:", data?.[0]);
    }
}

testQuery();
