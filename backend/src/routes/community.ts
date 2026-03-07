import { FastifyInstance } from 'fastify';
import { supabase } from '../supabase.js';

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

            // Check trust score - only trusted users can flag
            const { data: profile } = await supabase
                .from('profiles')
                .select('trust_score, is_community_verifier')
                .eq('id', user.id)
                .single();

            if (!profile || (profile.trust_score < 70 && !profile.is_community_verifier)) {
                return reply.code(403).send({
                    error: 'Insufficient trust score',
                    message: 'You need a trust score of 70+ or Community Verifier status to flag content'
                });
            }

            // Validate flag type
            const validTypes = ['MISINFORMATION', 'MANIPULATED', 'OUT_OF_CONTEXT', 'SPAM', 'OTHER'];
            if (!validTypes.includes(flagType)) {
                return reply.code(400).send({ error: 'Invalid flag type' });
            }

            // Insert flag
            const { data: flag, error: flagError } = await supabase
                .from('community_flags')
                .insert({
                    post_id: postId,
                    flagger_id: user.id,
                    flag_type: flagType,
                    reason: reason || null,
                    source_url: sourceUrl || null
                })
                .select()
                .single();

            if (flagError) {
                if (flagError.code === '23505') {
                    return reply.code(400).send({ error: 'You have already flagged this post' });
                }
                throw flagError;
            }

            // Check total flags - if >= 3, auto-notify post owner
            const { count: totalFlags } = await supabase
                .from('community_flags')
                .select('*', { count: 'exact', head: true })
                .eq('post_id', postId);

            if (totalFlags && totalFlags >= 3) {
                // Get post owner
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
                        message: `Your post has been flagged by ${totalFlags} community verifiers for review.`,
                        related_post_id: postId
                    });
                }
            }

            return { success: true, flag };
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
