import { supabase } from '../supabase.js';

/**
 * Recalculate and update a user's trust score based on:
 *   1. Verification Rate (0-40 pts)
 *   2. Account Age (0-20 pts)
 *   3. Activity Volume (0-20 pts)
 *   4. Community Reputation (0-20 pts)
 */
export async function updateProfileTrustScore(userId: string) {
  try {
    const { data: profile } = await supabase
      .from('profiles')
      .select('created_at')
      .eq('id', userId)
      .single();

    const [{ count: total }, { count: real }] = await Promise.all([
      supabase.from('verification_logs').select('*', { count: 'exact', head: true }).eq('user_id', userId),
      supabase.from('verification_logs').select('*', { count: 'exact', head: true }).eq('user_id', userId).in('final_verdict', ['REAL', 'APPROVED'])
    ]);

    const totalUploads = total || 0;
    const verifiedUploads = real || 0;
    const rejectedUploads = totalUploads - verifiedUploads;
    const realPercentage = totalUploads > 0 ? Math.round((verifiedUploads / totalUploads) * 100) : 100;
    const fakePercentage = totalUploads > 0 ? Math.round((rejectedUploads / totalUploads) * 100) : 0;

    // COMPONENT 1: Verification Rate (0-40 points)
    const verificationRate = totalUploads > 0 ? (verifiedUploads / totalUploads) : 0.5;
    const verificationScore = Math.round(verificationRate * 40);

    // COMPONENT 2: Account Age (0-20 points)
    const accountAgeDays = profile?.created_at
      ? (Date.now() - new Date(profile.created_at).getTime()) / (1000 * 86400)
      : 0;
    const ageScore = Math.min(20, Math.round((accountAgeDays / 90) * 20));

    // COMPONENT 3: Activity Volume (0-20 points)
    const volumeScore = Math.min(20, Math.round((totalUploads / 50) * 20));

    // COMPONENT 4: Community Reputation (0-20 points)
    const { count: followers } = await supabase
      .from('follows')
      .select('*', { count: 'exact', head: true })
      .eq('following_id', userId);
    const followerScore = Math.min(10, Math.round(((followers || 0) / 100) * 10));

    const { data: userPosts } = await supabase
      .from('posts')
      .select('like_count')
      .eq('user_id', userId);
    const totalLikeCount = userPosts?.reduce((sum: number, p: any) => sum + (p.like_count || 0), 0) || 0;
    const engagementScore = Math.min(10, Math.round((totalLikeCount / 500) * 10));
    const communityScore = followerScore + engagementScore;

    // FINAL TRUST SCORE (0-100)
    const trustScore = Math.min(100, verificationScore + ageScore + volumeScore + communityScore);

    let status = 'TRUSTED';
    if (totalUploads === 0) status = 'NEW_USER';
    else if (trustScore < 25) status = 'UNDER_REVIEW';
    else if (trustScore < 50) status = 'AT_RISK';

    await supabase.from('profiles').update({
      trust_status: status,
      trust_score: trustScore,
      trust_score_updated_at: new Date().toISOString(),
      real_percentage: realPercentage,
      fake_percentage: fakePercentage,
      total_attempts: totalUploads,
      real_count: verifiedUploads,
      fake_count: rejectedUploads
    }).eq('id', userId);

    await supabase.from('trust_score_history').insert({
      user_id: userId,
      trust_score: trustScore
    });

    console.log(`[TRUST-CACHE] User ${userId}: ${status} (Score: ${trustScore}, ${realPercentage}% Real)`);
  } catch (e) { console.warn(`[TRUST-CACHE] Failed for ${userId}`, e); }
}

/**
 * Track a deepfake detection in the alerts system.
 */
export async function trackDeepfakeAlert(mediaHash: string, reason: string) {
  try {
    const { data: existing } = await supabase
      .from('deepfake_alerts')
      .select('id, detection_count')
      .eq('media_hash', mediaHash)
      .single();

    if (existing) {
      await supabase.from('deepfake_alerts')
        .update({
          detection_count: existing.detection_count + 1,
          severity: existing.detection_count >= 5 ? 'CRITICAL' : existing.detection_count >= 3 ? 'HIGH' : 'MEDIUM',
          updated_at: new Date().toISOString()
        })
        .eq('id', existing.id);
    } else {
      await supabase.from('deepfake_alerts').insert({
        title: 'Deepfake Content Detected',
        description: reason,
        media_hash: mediaHash,
        severity: 'LOW'
      });
    }
  } catch (e) {
    console.warn('[ALERT] Deepfake alert tracking failed:', e);
  }
}
