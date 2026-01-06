import { supabase } from './supabase.js';

async function syncAllProfilesDetailed() {
    console.log('[SYNC] Starting detailed profile trust score synchronization...');

    const { data: profiles, error: pError } = await supabase.from('profiles').select('id, username');
    if (pError) {
        console.error('Failed to fetch profiles:', pError);
        return;
    }

    for (const profile of profiles) {
        console.log(`[SYNC] Processing ${profile.username}...`);

        const [{ count: total }, { count: real }] = await Promise.all([
            supabase.from('verification_logs').select('*', { count: 'exact', head: true }).eq('user_id', profile.id),
            supabase.from('verification_logs').select('*', { count: 'exact', head: true }).eq('user_id', profile.id).in('final_verdict', ['REAL', 'APPROVED'])
        ]);

        const totalUploads = total || 0;
        const verifiedUploads = real || 0;
        const rejectedUploads = totalUploads - verifiedUploads;
        const realPercentage = totalUploads > 0 ? Math.round((verifiedUploads / totalUploads) * 100) : 100;
        const fakePercentage = totalUploads > 0 ? Math.round((rejectedUploads / totalUploads) * 100) : 0;

        let status = 'TRUSTED';
        if (totalUploads === 0) status = 'NEW_USER';
        else if (fakePercentage > 30) status = 'RESTRICTED';
        else if (fakePercentage > 10) status = 'AT_RISK';

        // Update with all new columns
        const updateData: any = {
            trust_status: status,
            real_percentage: realPercentage,
            fake_percentage: fakePercentage,
            total_attempts: totalUploads,
            real_count: verifiedUploads,
            fake_count: rejectedUploads
        };

        const { error: uError } = await supabase.from('profiles').update(updateData).eq('id', profile.id);

        if (uError) {
            console.warn(`[SYNC] Could not update granular columns for ${profile.username} (Migration might be missing):`, uError.message);
            // Fallback to minimal update if migration is missing
            await supabase.from('profiles').update({
                trust_status: status,
                real_percentage: realPercentage
            }).eq('id', profile.id);
        } else {
            console.log(`[SYNC] Updated ${profile.username}: ${status} (${realPercentage}% Real, ${fakePercentage}% Fake)`);
        }
    }

    console.log('[SYNC] Completed.');
}

syncAllProfilesDetailed();
