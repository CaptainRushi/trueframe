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
                        bio
                    ),
                    verification:verification_log_id (
                        verdict,
                        score,
                        created_at
                    )
                `)
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

            // 2. Fetch Trust Metrics for all authors in this list
            const authorIds = [...new Set(rawPosts.map((p: any) => p.user_id))];

            // We'll calculate a "Trust Score" for each author
            // In a production app, we'd pre-calculate this or use an RPC
            // For now, we'll fetch verification counts
            const authorTrustMap: Record<string, number> = {};

            for (const id of authorIds) {
                const { count: total } = await supabase
                    .from('verification_logs')
                    .select('*', { count: 'exact', head: true })
                    .eq('user_id', id);

                const { count: fake } = await supabase
                    .from('verification_logs')
                    .select('*', { count: 'exact', head: true })
                    .eq('user_id', id)
                    .in('verdict', ['FAKE', 'REJECTED', 'BLOCK_FAKE', 'BLOCK_UNVERIFIED', 'BLOCK_ERROR']);

                const totalVal = total || 0;
                const fakeVal = fake || 0;
                const fakeRate = totalVal > 0 ? (fakeVal / totalVal) : 0;

                // Trust Status: Trusted (1.0), At Risk (0.5), Restricted (0.1)
                let trustWeight = 1.0;
                if (fakeRate > 0.3) trustWeight = 0.1;
                else if (fakeRate > 0.1) trustWeight = 0.5;

                authorTrustMap[id as string] = trustWeight;
            }

            const rankedPosts = rawPosts.map((post: any) => {
                const trustWeight = authorTrustMap[post.user_id as string] || 0.5;
                const hoursOld = (Date.now() - new Date(post.created_at).getTime()) / (1000 * 3600);
                const recencyScore = Math.max(0, 1 - (hoursOld / 72)); // 0 if > 3 days old

                return {
                    ...post,
                    rankingScore: (0.7 * trustWeight) + (0.3 * recencyScore)
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
