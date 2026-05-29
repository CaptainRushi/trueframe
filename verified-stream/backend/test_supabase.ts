import { supabase } from './src/supabase.js';

async function test() {
    try {
        console.log("Empty string test");
        const res1 = await supabase.auth.getUser('');
        console.log("res1", res1);
    } catch(e) {
        console.error("Test 1 threw:", e);
    }
    
    try {
        console.log("Null token test");
        const res2 = await supabase.auth.getUser(null as any);
        console.log("res2", res2);
    } catch(e) {
        console.error("Test 2 threw:", e);
    }
}
test();
