import { FastifyInstance } from 'fastify';
import { supabase } from '../supabase.js';
import { triggerSecondaryReview } from './moderation.js';

export async function communityRoutes(fastify: FastifyInstance) {

    /**
     * POST /api/community/flag/:postId
     * Community Fact-Checking - Flag a post
     * Only users with trust_score >= 70 can flag
     */
    fastify.post<{ Params: { postId: string } }>('/flag/:postId', async (request, reply) => {
        const { postId } = request.params;
        const { flagType, reason, sourceUrl } = request.body as {
            flagType: string;
            reason?: string;
            sourceUrl?: string;
        };
        const authHeader = request.headers.authorization;
        if (!authHeader) return reply.code(401).send({ error: 'Unauthorized' });

        try {
            const token = authHeader.replace('Bearer ', '');
            const { data: { user }, error: authError } = await supabase.auth.getUser(token);
            if (authError || !user) return reply.code(401).send({ error: 'Invalid token' });

            // Fetch profile for flag weight calculation
            const { data: profile } = await supabase
                .from('profiles')
                .select('trust_score, is_community_verifier')
                .eq('id', user.id)
                .single();

            const trustScore = profile?.trust_score ?? 50;
            const isVerifier = profile?.is_community_verifier || false;

            // Compute flag weight based on flagger's trust score (low score = low weight, but flag is saved)
            const flagWeight = Math.max(0.05, trustScore / 100);

            // Validate flag type
            const validTypes = ['MISINFORMATION', 'MANIPULATED', 'OUT_OF_CONTEXT', 'SPAM', 'OTHER'];
            if (!validTypes.includes(flagType)) {
                return reply.code(400).send({ error: 'Invalid flag type' });
            }

            // Insert flag with weight
            const { data: flag, error: flagError } = await supabase
                .from('community_flags')
                .insert({
                    post_id: postId,
                    flagger_id: user.id,
                    flag_type: flagType,
                    reason: reason || null,
                    source_url: sourceUrl || null,
                    flag_weight: flagWeight
                })
                .select()
                .single();

            if (flagError) {
                if (flagError.code === '23505') {
                    return reply.code(400).send({ error: 'You have already flagged this post' });
                }
                throw flagError;
            }

            // Compute weighted flag sum for this post
            const { data: allFlags } = await supabase
                .from('community_flags')
                .select('flag_weight')
                .eq('post_id', postId)
                .eq('status', 'PENDING');

            const totalFlags = allFlags?.length || 0;
            const weightedSum = allFlags?.reduce((sum: number, f: any) => sum + (f.flag_weight || 1.0), 0) || 0;

            // Check if secondary review was previously restored (higher threshold to prevent re-harassment)
            const { data: existingReview } = await supabase
                .from('secondary_reviews')
                .select('id, decision')
                .eq('post_id', postId)
                .maybeSingle();

            const wasRestored = existingReview?.decision === 'RESTORE';
            const threshold = wasRestored ? 4.0 : 2.5;
            const alreadyUnderReview = existingReview && !wasRestored;
            let reviewTriggered = false;

            if (weightedSum >= threshold && !alreadyUnderReview) {
                // Threshold reached and no existing review — trigger secondary review
                reviewTriggered = true;

                // Set post to UNDER_REVIEW
                await supabase.from('posts')
                    .update({ visibility: 'UNDER_REVIEW', verification_status: 'UNDER_REVIEW' })
                    .eq('id', postId);

                // Mark flags as having triggered a review
                await supabase.from('community_flags')
                    .update({ triggered_review: true })
                    .eq('post_id', postId)
                    .eq('status', 'PENDING');

                // Create or reset secondary review record
                if (wasRestored && existingReview) {
                    // Reset the previous RESTORE review for re-analysis
                    await supabase.from('secondary_reviews')
                        .update({
                            triggered_by: 'COMMUNITY_FLAGS',
                            trigger_flag_count: totalFlags,
                            status: 'PENDING',
                            secondary_score: null,
                            frequency_score: null,
                            gan_artifact_score: null,
                            noise_consistency_score: null,
                            edge_coherence_score: null,
                            patch_variance_score: null,
                            score_breakdown: null,
                            signals: null,
                            decision: null,
                            decision_reason: null,
                            manual_reviewer_id: null,
                            manual_decision: null,
                            manual_review_reason: null,
                            manual_reviewed_at: null,
                            completed_at: null,
                            created_at: new Date().toISOString()
                        })
                        .eq('id', existingReview.id);
                } else {
                    await supabase.from('secondary_reviews').insert({
                        post_id: postId,
                        triggered_by: 'COMMUNITY_FLAGS',
                        trigger_flag_count: totalFlags,
                        status: 'PENDING'
                    });
                }

                // Notify post owner
                const { data: post } = await supabase
                    .from('posts')
                    .select('user_id')
                    .eq('id', postId)
                    .single();

                if (post) {
                    await supabase.from('notifications').insert({
                        user_id: post.user_id,
                        type: 'FLAG_RESULT',
                        title: 'Content Under Review',
                        message: `Your post has been flagged by ${totalFlags} community members and is now under secondary AI review.`,
                        related_post_id: postId
                    });
                }

                // Fire-and-forget secondary review
                triggerSecondaryReview(postId).catch((err: any) => {
                    console.error(`[COMMUNITY] Secondary review trigger failed for ${postId}:`, err);
                });
            } else if (totalFlags >= 3 && !existingReview) {
                // Below weighted threshold but 3+ flags — still notify owner
                const { data: post } = await supabase
                    .from('posts')
                    .select('user_id')
                    .eq('id', postId)
                    .single();

                if (post) {
                    await supabase.from('notifications').insert({
                        user_id: post.user_id,
                        type: 'FLAG_RESULT',
                        title: 'Content Flagged',
                        message: `Your post has been flagged by ${totalFlags} community members for review.`,
                        related_post_id: postId
                    });
                }
            }

            return { success: true, flag, reviewTriggered };
        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to flag post' });
        }
    });

    /**
     * GET /api/community/flags/:postId
     * Get flags for a post
     */
    fastify.get<{ Params: { postId: string } }>('/flags/:postId', async (request, reply) => {
        const { postId } = request.params;

        try {
            const { data: flags, error } = await supabase
                .from('community_flags')
                .select(`
                    id, flag_type, reason, source_url, status, created_at,
                    flagger:flagger_id (
                        username, display_name, trust_score, is_community_verifier
                    )
                `)
                .eq('post_id', postId)
                .order('created_at', { ascending: false });

            if (error) throw error;

            return { flags: flags || [] };
        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to fetch flags' });
        }
    });

    /**
     * POST /api/community/flag/:flagId/review
     * Review a community flag (admin/high-trust only)
     */
    fastify.post<{ Params: { flagId: string } }>('/flag/:flagId/review', async (request, reply) => {
        const { flagId } = request.params;
        const { status } = request.body as { status: 'CONFIRMED' | 'DISMISSED' };
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

            if (!profile || (profile.trust_score < 90 && !profile.is_community_verifier)) {
                return reply.code(403).send({ error: 'Only senior verifiers can review flags' });
            }

            const { error } = await supabase
                .from('community_flags')
                .update({
                    status,
                    reviewed_by: user.id,
                    reviewed_at: new Date().toISOString()
                })
                .eq('id', flagId);

            if (error) throw error;

            return { success: true };
        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to review flag' });
        }
    });

    /**
     * GET /api/community/verifiers
     * List community verifiers (high-trust users eligible for verification)
     */
    fastify.get('/verifiers', async (request, reply) => {
        try {
            const { data: verifiers, error } = await supabase
                .from('profiles')
                .select('id, username, display_name, avatar_url, trust_score, is_community_verifier, is_verified_creator')
                .or('is_community_verifier.eq.true,trust_score.gte.80')
                .order('trust_score', { ascending: false })
                .limit(50);

            if (error) throw error;
            return { verifiers: verifiers || [] };
        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to fetch verifiers' });
        }
    });
}
