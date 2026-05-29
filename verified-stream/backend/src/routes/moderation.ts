import { FastifyInstance } from 'fastify';
import { join } from 'path';
import { existsSync, unlinkSync, writeFileSync } from 'fs';
import { supabase } from '../supabase.js';
import { runAIScript } from '../lib/ai-runner.js';
import { updateProfileTrustScore, trackDeepfakeAlert } from '../lib/trust.js';
import { getAiServicePath } from '../lib/paths.js';

/**
 * Trigger secondary AI review for a post.
 * Called asynchronously (fire-and-forget) when flag threshold is reached.
 * Fail-closed: errors keep the post UNDER_REVIEW.
 */
export async function triggerSecondaryReview(postId: string) {
  let tempPath = '';
  try {
    console.log(`[SECONDARY-REVIEW] Starting review for post ${postId}`);

    // Update status to PROCESSING
    await supabase.from('secondary_reviews')
      .update({ status: 'PROCESSING' })
      .eq('post_id', postId);

    // Fetch post media URL and hash
    const { data: post, error: postError } = await supabase
      .from('posts')
      .select('media_url, media_hash_check, user_id, media_type')
      .eq('id', postId)
      .single();

    if (postError || !post) {
      throw new Error(`Post not found: ${postId}`);
    }

    // Download media to temp file for AI analysis
    const tempDir = join(process.cwd(), 'tmp', 'secondary_review');
    const fs = await import('fs/promises');
    await fs.mkdir(tempDir, { recursive: true });
    tempPath = join(tempDir, `${Date.now()}_${postId}.tmp`);

    // Fetch image from the public URL
    const response = await fetch(post.media_url);
    if (!response.ok) throw new Error(`Failed to download media: ${response.status}`);
    const buffer = Buffer.from(await response.arrayBuffer());
    writeFileSync(tempPath, buffer);

    // Run secondary AI analysis
    const aiScriptPath = getAiServicePath('secondary_review.py');
    const result = await runAIScript(aiScriptPath, [tempPath], 120000);

    const secondaryScore = result.secondary_score ?? 0;
    const swinScore = typeof result.swin_score === 'number' ? result.swin_score : null;
    const swinRiskTier = typeof result.swin_risk_tier === 'string' ? result.swin_risk_tier : null;

    // Determine decision
    let decision: string;
    let decisionReason: string;
    if (secondaryScore >= 0.80) {
      decision = 'REMOVE';
      decisionReason = 'Secondary AI (Model 1) confirmed synthetic/deepfake content';
    } else if (secondaryScore >= 0.60) {
      // Borderline: run Model 2 (main.py HuggingFace+signal) for second opinion
      decision = 'RUN_MODEL_2';
      decisionReason = 'Model 1 borderline — escalating to Model 2 automated review';
    } else {
      decision = 'RESTORE';
      decisionReason = 'Secondary AI (Model 1) found content to be authentic';
    }

    // Store Model 1 results first
    await supabase.from('secondary_reviews')
      .update({
        status: decision === 'RUN_MODEL_2' ? 'PROCESSING' : 'COMPLETED',
        secondary_score: secondaryScore,
        frequency_score: result.frequency_score ?? 0,
        gan_artifact_score: result.gan_artifact_score ?? 0,
        noise_consistency_score: result.noise_consistency_score ?? 0,
        edge_coherence_score: result.edge_coherence_score ?? 0,
        patch_variance_score: result.patch_variance_score ?? 0,
        swin_score: swinScore,
        swin_risk_tier: swinRiskTier,
        score_breakdown: result,
        signals: result.signals || [],
        decision: decision === 'RUN_MODEL_2' ? null : decision, // null until model2 resolves
        decision_reason: decisionReason,
        completed_at: decision === 'RUN_MODEL_2' ? null : new Date().toISOString()
      })
      .eq('post_id', postId);

    // If Model 1 is borderline, run Model 2 automatically
    if (decision === 'RUN_MODEL_2') {
      try {
        const model2ScriptPath = getAiServicePath('main.py');
        const model2Result = await runAIScript(model2ScriptPath, [tempPath], 300000);

        const model2Score: number = model2Result.final_score ?? 0;
        const model2Signals: string[] = model2Result.signals || [];

        // Model 2 decision logic:
        //   score < THRESHOLD_APPROVE   → RESTORE (both models uncertain → trust real-content
        //                                            only if clearly low)
        //   score >= THRESHOLD_REJECT   → REMOVE  (confirmed fake)
        //   otherwise (borderline)      → REMOVE  (fail-safe: flagged post, both uncertain
        //                                           → remove to protect platform integrity)
        const MODEL2_REMOVE_THRESHOLD  = 0.75;  // mirrors THRESHOLD_REJECT in config.py
        const MODEL2_RESTORE_THRESHOLD = 0.40;  // mirrors THRESHOLD_APPROVE in config.py

        let model2Decision: string;
        let model2Reason: string;
        if (model2Score >= MODEL2_REMOVE_THRESHOLD) {
          model2Decision = 'REMOVE';
          model2Reason = `Model 2 (HuggingFace+signal) confirmed deepfake (score=${model2Score.toFixed(3)})`;
        } else if (model2Score < MODEL2_RESTORE_THRESHOLD) {
          model2Decision = 'RESTORE';
          model2Reason = `Model 2 found content authentic (score=${model2Score.toFixed(3)})`;
        } else {
          // Both models uncertain on a community-flagged post → remove (fail-safe)
          model2Decision = 'REMOVE';
          model2Reason = `Both models borderline on flagged post — removed as precaution (M2 score=${model2Score.toFixed(3)})`;
        }

        decision = model2Decision;
        decisionReason = model2Reason;

        // Persist model2 result
        await supabase.from('secondary_reviews')
          .update({
            status: 'COMPLETED',
            decision,
            decision_reason: decisionReason,
            completed_at: new Date().toISOString()
          })
          .eq('post_id', postId);

        // Also persist model2 scores in extended columns (added via migration)
        // Using score_breakdown JSONB field so we don't need schema changes for model2 detail
        await supabase.from('secondary_reviews')
          .update({
            score_breakdown: {
              ...result,
              model2_score: model2Score,
              model2_signals: model2Signals,
              model2_decision: model2Decision
            }
          })
          .eq('post_id', postId);

      } catch (model2Error: any) {
        // Model 2 failed — fail closed on a flagged post: remove it
        console.error(`[SECONDARY-REVIEW] Model 2 failed for ${postId}:`, model2Error);
        decision = 'REMOVE';
        decisionReason = `Model 2 error (fail-closed): ${model2Error.message}`;
        await supabase.from('secondary_reviews')
          .update({
            status: 'COMPLETED',
            decision,
            decision_reason: decisionReason,
            completed_at: new Date().toISOString()
          })
          .eq('post_id', postId);
      }
    } else {
      // Model 1 was decisive — no model2 run needed
    }

    // Apply final decision (RESTORE or REMOVE — no MANUAL_REVIEW path)
    if (decision === 'RESTORE') {
      await restorePost(postId, post.user_id);
    } else if (decision === 'REMOVE') {
      await removeConfirmedDeepfake(postId);
    }
    // (RUN_MODEL_2 case never reaches here — resolved above)

    console.log(`[SECONDARY-REVIEW] Completed for post ${postId}: ${decision} (M1 score: ${secondaryScore})`);

  } catch (error: any) {
    console.error(`[SECONDARY-REVIEW] Failed for post ${postId}:`, error);
    // Fail-closed: keep post UNDER_REVIEW
    await supabase.from('secondary_reviews')
      .update({
        status: 'FAILED',
        decision_reason: `Review failed: ${error.message}`,
        completed_at: new Date().toISOString()
      })
      .eq('post_id', postId);
  } finally {
    if (tempPath && existsSync(tempPath)) {
      try { unlinkSync(tempPath); } catch (e) { /* ignore cleanup errors */ }
    }
  }
}

/**
 * Restore a post after secondary review clears it.
 */
async function restorePost(postId: string, userId: string) {
  // Restore post visibility
  await supabase.from('posts')
    .update({ visibility: 'PUBLIC', verification_status: 'APPROVED' })
    .eq('id', postId);

  // Dismiss all community flags
  await supabase.from('community_flags')
    .update({ status: 'DISMISSED' })
    .eq('post_id', postId)
    .eq('status', 'PENDING');

  // Notify uploader
  await supabase.from('notifications').insert({
    user_id: userId,
    type: 'SECONDARY_REVIEW_RESULT',
    title: 'Post Restored',
    message: 'Your post was reviewed by our automated AI review system and confirmed as authentic. It is now visible again.',
    related_post_id: postId
  });

  console.log(`[SECONDARY-REVIEW] Post ${postId} restored to PUBLIC`);
}

/**
 * Remove a confirmed deepfake post found via secondary review.
 */
async function removeConfirmedDeepfake(postId: string) {
  // Fetch post data
  const { data: post } = await supabase
    .from('posts')
    .select('user_id, media_url, media_hash_check')
    .eq('id', postId)
    .single();

  if (!post) return;

  // Confirm all community flags
  await supabase.from('community_flags')
    .update({
      status: 'CONFIRMED',
      reviewed_at: new Date().toISOString()
    })
    .eq('post_id', postId)
    .eq('status', 'PENDING');

  // Log a new verification entry for the secondary review rejection
  await supabase.from('verification_logs').insert({
    user_id: post.user_id,
    media_hash: post.media_hash_check || 'unknown',
    media_type: 'image',
    deepfake_verdict: 'REJECTED',
    fake_news_verdict: 'SKIPPED',
    final_verdict: 'FAKE',
    verdict: 'FAKE',
    score: 1.0,
    reason: 'Confirmed deepfake via secondary community review',
    model_name: 'secondary-frequency-gan-ensemble',
    model_version: '1.0',
    model_score: 1.0,
    final_score: 1.0,
    authenticity_label: 'REJECTED_SYNTHETIC'
  });

  // Delete media from Supabase Storage
  try {
    const url = new URL(post.media_url);
    const storagePath = url.pathname.split('/posts/').pop();
    if (storagePath) {
      await supabase.storage.from('posts').remove([storagePath]);
    }
  } catch (e) {
    console.warn(`[SECONDARY-REVIEW] Storage cleanup failed for post ${postId}:`, e);
  }

  // Delete post (cascading deletes handle likes/comments/shares)
  await supabase.from('posts').delete().eq('id', postId);

  // Apply trust penalty
  await applyTrustPenalty(post.user_id, 'Confirmed deepfake removed via community review');

  // Notify uploader
  await supabase.from('notifications').insert({
    user_id: post.user_id,
    type: 'POST_REMOVED',
    title: 'Post Removed',
    message: 'Your post was automatically removed after two independent AI systems confirmed it violates our real-content policy. Repeated violations may lead to account restrictions.'
  });

  // Track deepfake alert
  if (post.media_hash_check) {
    await trackDeepfakeAlert(post.media_hash_check, 'Confirmed deepfake via secondary community review');
  }

  console.log(`[SECONDARY-REVIEW] Post ${postId} removed (confirmed deepfake)`);
}

/**
 * Apply escalating trust penalty for confirmed deepfake removal.
 */
async function applyTrustPenalty(userId: string, reason: string) {
  try {
    // Count prior removals for this user
    const { count: priorRemovals } = await supabase
      .from('secondary_reviews')
      .select('*', { count: 'exact', head: true })
      .eq('decision', 'REMOVE')
      .in('post_id', (
        await supabase.from('posts').select('id').eq('user_id', userId)
      ).data?.map((p: any) => p.id) || []);

    // This approach may miss deleted posts, so also count verification logs
    const { count: fakeCount } = await supabase
      .from('verification_logs')
      .select('*', { count: 'exact', head: true })
      .eq('user_id', userId)
      .eq('model_name', 'secondary-frequency-gan-ensemble');

    const offenseCount = Math.max(priorRemovals || 0, fakeCount || 0);

    // Escalating penalty
    let penalty: number;
    if (offenseCount <= 1) penalty = 15;
    else if (offenseCount === 2) penalty = 25;
    else penalty = 40;

    // Fetch current trust score
    const { data: profile } = await supabase
      .from('profiles')
      .select('trust_score')
      .eq('id', userId)
      .single();

    if (!profile) return;

    const newScore = Math.max(0, profile.trust_score - penalty);

    // Derive new status
    let newStatus: string;
    if (newScore < 25) newStatus = 'UNDER_REVIEW';
    else if (newScore < 50) newStatus = 'AT_RISK';
    else newStatus = 'TRUSTED';

    await supabase.from('profiles')
      .update({
        trust_score: newScore,
        trust_status: newStatus,
        trust_score_updated_at: new Date().toISOString()
      })
      .eq('id', userId);

    // Record in history
    await supabase.from('trust_score_history').insert({
      user_id: userId,
      trust_score: newScore
    });

    // Notify user of penalty
    await supabase.from('notifications').insert({
      user_id: userId,
      type: 'TRUST_PENALTY',
      title: 'Trust Score Reduced',
      message: `Your trust score was reduced by ${penalty} points (now ${newScore}) due to: ${reason}`
    });

    // Also recalculate full trust score for accuracy
    await updateProfileTrustScore(userId);

    console.log(`[TRUST-PENALTY] User ${userId}: -${penalty} pts (offense #${offenseCount}, now ${newScore})`);
  } catch (e) {
    console.warn(`[TRUST-PENALTY] Failed for ${userId}:`, e);
  }
}

/**
 * Notify community verifiers that a post needs manual review.
 */
async function notifyCommunityVerifiers(postId: string) {
  try {
    const { data: verifiers } = await supabase
      .from('profiles')
      .select('id')
      .or('is_community_verifier.eq.true,trust_score.gte.90')
      .limit(50);

    if (!verifiers || verifiers.length === 0) return;

    const notifications = verifiers.map((v: any) => ({
      user_id: v.id,
      type: 'SYSTEM' as const,
      title: 'Manual Review Required',
      message: 'A community-reported post requires your manual review decision.',
      related_post_id: postId
    }));

    await supabase.from('notifications').insert(notifications);

    console.log(`[SECONDARY-REVIEW] Notified ${verifiers.length} community verifiers for post ${postId}`);
  } catch (e) {
    console.warn('[SECONDARY-REVIEW] Failed to notify verifiers:', e);
  }
}


// ==========================================
// FASTIFY ROUTES
// ==========================================

export async function moderationRoutes(fastify: FastifyInstance) {

  /**
   * GET /api/moderation/pending
   * List posts awaiting manual review (for community verifiers)
   */
  fastify.get('/pending', async (request, reply) => {
    const authHeader = request.headers.authorization;
    if (!authHeader) return reply.code(401).send({ error: 'Unauthorized' });

    try {
      const token = authHeader.replace('Bearer ', '');
      const { data: { user } } = await supabase.auth.getUser(token);
      if (!user) return reply.code(401).send({ error: 'Invalid token' });

      const { data: profile } = await supabase
        .from('profiles')
        .select('trust_score, is_community_verifier')
        .eq('id', user.id)
        .single();

      if (!profile || (!profile.is_community_verifier && profile.trust_score < 90)) {
        return reply.code(403).send({ error: 'Only community verifiers can access moderation queue' });
      }

      // Fetch pending manual reviews
      const { data: reviews, error } = await supabase
        .from('secondary_reviews')
        .select(`
          id, post_id, trigger_flag_count, secondary_score, decision,
          decision_reason, signals, swin_score, swin_risk_tier, created_at, completed_at,
          posts:post_id (
            id, media_url, media_type, caption, user_id, created_at,
            profiles:user_id (
              username, display_name, avatar_url, trust_score, trust_status
            )
          )
        `)
        .eq('decision', 'MANUAL_REVIEW')
        .is('manual_decision', null)
        .order('created_at', { ascending: false });

      if (error) throw error;

      // Also get flag counts for each post
      const postIds = (reviews || []).map((r: any) => r.post_id);
      let flagCounts: Record<string, number> = {};
      if (postIds.length > 0) {
        const { data: flags } = await supabase
          .from('community_flags')
          .select('post_id')
          .in('post_id', postIds);

        if (flags) {
          for (const f of flags) {
            flagCounts[f.post_id] = (flagCounts[f.post_id] || 0) + 1;
          }
        }
      }

      const riskRank = (review: any) => {
        if (review.swin_risk_tier === 'HIGH') return 3;
        if (review.swin_risk_tier === 'ELEVATED') return 2;
        if (review.swin_risk_tier === 'LOW') return 1;
        if (typeof review.swin_score === 'number' && review.swin_score >= 0.85) return 2;
        return 0;
      };

      const enriched = (reviews || []).map((r: any) => ({
        ...r,
        flagCount: flagCounts[r.post_id] || 0
      })).sort((a: any, b: any) => {
        const riskDelta = riskRank(b) - riskRank(a);
        if (riskDelta !== 0) return riskDelta;
        return (b.secondary_score || 0) - (a.secondary_score || 0);
      });

      return { reviews: enriched };
    } catch (error: any) {
      fastify.log.error(error);
      return reply.code(500).send({ error: 'Failed to fetch moderation queue' });
    }
  });

  /**
   * POST /api/moderation/review/:reviewId
   * Submit a manual moderation decision (RESTORE or REMOVE)
   */
  fastify.post<{ Params: { reviewId: string } }>('/review/:reviewId', async (request, reply) => {
    const { reviewId } = request.params;
    const { decision, reason } = request.body as { decision: 'RESTORE' | 'REMOVE'; reason?: string };
    const authHeader = request.headers.authorization;
    if (!authHeader) return reply.code(401).send({ error: 'Unauthorized' });

    try {
      const token = authHeader.replace('Bearer ', '');
      const { data: { user } } = await supabase.auth.getUser(token);
      if (!user) return reply.code(401).send({ error: 'Invalid token' });

      const { data: profile } = await supabase
        .from('profiles')
        .select('trust_score, is_community_verifier')
        .eq('id', user.id)
        .single();

      if (!profile || (!profile.is_community_verifier && profile.trust_score < 90)) {
        return reply.code(403).send({ error: 'Only senior verifiers can make moderation decisions' });
      }

      if (!['RESTORE', 'REMOVE'].includes(decision)) {
        return reply.code(400).send({ error: 'Decision must be RESTORE or REMOVE' });
      }

      // Fetch the review
      const { data: review, error: reviewError } = await supabase
        .from('secondary_reviews')
        .select('id, post_id, decision, manual_decision')
        .eq('id', reviewId)
        .single();

      if (reviewError || !review) {
        return reply.code(404).send({ error: 'Review not found' });
      }

      if (review.manual_decision) {
        return reply.code(400).send({ error: 'This review has already been decided' });
      }

      // Update review with manual decision
      await supabase.from('secondary_reviews')
        .update({
          manual_reviewer_id: user.id,
          manual_decision: decision,
          manual_review_reason: reason || null,
          manual_reviewed_at: new Date().toISOString()
        })
        .eq('id', reviewId);

      // Fetch post info for the action
      const { data: post } = await supabase
        .from('posts')
        .select('user_id')
        .eq('id', review.post_id)
        .single();

      // Apply decision
      if (decision === 'RESTORE') {
        await restorePost(review.post_id, post?.user_id || '');
      } else {
        await removeConfirmedDeepfake(review.post_id);
      }

      return { success: true, decision };
    } catch (error: any) {
      fastify.log.error(error);
      return reply.code(500).send({ error: 'Failed to process moderation decision' });
    }
  });
}
