import { FastifyInstance } from 'fastify';
import { supabase } from '../supabase.js';

export async function feedRoutes(fastify: FastifyInstance) {

    // FEATURE 6: Trust-Weighted Discovery
    fastify.get('/', async (request, reply) => {
        try {
            console.log('[FEED-API] Fetching posts...');

            // 1. Fetch Posts with Author and Verification Data
            const { data: rawPosts, error } = await supabase
                .from('posts')
                .select(`
                    *,
                    profiles:user_id!inner (
                        id,
                        username,
                        display_name,
                        avatar_url,
                        bio,
                        trust_score,
                        trust_status,
                        is_verified_creator
                    ),
                    verification:verification_log_id (
                        verdict,
                        score,
                        created_at,
                        authenticity_label,
                        score_breakdown
                    )
                `)
                .eq('visibility', 'PUBLIC')
                .order('created_at', { ascending: false })
                .limit(50);

            if (error) {
                console.error('[FEED-API] Query error:', error);
                throw error;
            }

            if (!rawPosts || rawPosts.length === 0) {
                console.log('[FEED-API] No posts found, returning empty array');
                return { posts: [] };
            }

            console.log(`[FEED-API] Found ${rawPosts.length} posts`);

            const rankedPosts = rawPosts.map((post: any) => {
                const profile = post.profiles || {};

                // Trust Status: Trusted (1.0), At Risk (0.5), Under Review (0.1)
                let trustWeight = 1.0;
                if (profile.trust_status === 'UNDER_REVIEW' || profile.trust_status === 'RESTRICTED') trustWeight = 0.1;
                else if (profile.trust_status === 'AT_RISK') trustWeight = 0.5;

                const hoursOld = (Date.now() - new Date(post.created_at).getTime()) / (1000 * 3600);
                const recencyScore = Math.max(0, 1 - (hoursOld / 72)); // 0 if > 3 days old

                // Boost verified creators
                const creatorBoost = post.profiles?.is_verified_creator ? 0.15 : 0;
                // Boost camera-captured content
                const cameraBoost = post.upload_source === 'CAMERA' ? 0.05 : 0;

                return {
                    ...post,
                    rankingScore: (0.6 * trustWeight) + (0.25 * recencyScore) + creatorBoost + cameraBoost
                };
            }).sort((a: any, b: any) => b.rankingScore - a.rankingScore);

            console.log(`[FEED-API] Returning ${rankedPosts.length} ranked posts`);

            return {
                posts: rankedPosts
            };

        } catch (error: any) {
            console.error('[FEED-API] Fatal error:', error);
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to fetch feed', posts: [] });
        }
    });
}
