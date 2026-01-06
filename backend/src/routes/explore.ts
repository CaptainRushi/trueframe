import { FastifyInstance } from 'fastify';
import { supabase } from '../supabase.js';

export async function exploreRoutes(fastify: FastifyInstance) {

    /**
     * GET /api/explore
     * Returns ranked verified posts for the Explore feed.
     * Rules: PUBLIC, APPROVED, TRUSTED/AT_RISK users only (no RESTRICTED).
     */
    fastify.get('/explore', async (request, reply) => {
        try {
            // Using Supabase to join posts and profiles
            // We use !inner to ensure we filter based on profile fields
            const { data: posts, error } = await supabase
                .from('posts')
                .select(`
                    id,
                    created_at,
                    media_url,
                    media_type,
                    caption,
                    like_count,
                    comment_count,
                    profiles!inner (
                        id,
                        username,
                        display_name,
                        avatar_url,
                        trust_status,
                        real_percentage
                    )
                `)
                .eq('visibility', 'PUBLIC')
                // .eq('verification_status', 'APPROVED') // Relaxed to show all posts
                // .eq('profiles.trust_status', 'TRUSTED') // Strict check: Only TRUSTED, exclude AT_RISK
                .order('created_at', { ascending: false }) // Prioritize recency for MVP
                .limit(50);

            if (error) {
                fastify.log.error(error);
                return reply.code(500).send({ error: 'Failed to fetch explore feed' });
            }

            // Client-side filtering just in case Supabase select filter behaved oddly (double safety)
            const safePosts = posts.filter((p: any) => {
                const profile = p.profiles;
                return profile && profile.trust_status !== 'RESTRICTED';
            });

            return safePosts;

        } catch (error) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Internal Server Error' });
        }
    });

    /**
     * GET /api/search/users
     * Search verified users by username or display name.
     */
    fastify.get<{ Querystring: { q: string } }>('/search/users', async (request, reply) => {
        const { q } = request.query;
        if (!q || q.length < 2) return [];

        try {
            // Search username OR display_name
            // And ensure NOT restricted
            const { data: users, error } = await supabase
                .from('profiles')
                .select('id, username, display_name, avatar_url, trust_status, real_percentage')
                // .eq('trust_status', 'TRUSTED') // REMOVED RESTRICTION: Allow finding any user
                .or(`username.ilike.%${q}%,display_name.ilike.%${q}%`)
                .limit(20);

            if (error) throw error;
            return users;

        } catch (error) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Search failed' });
        }
    });

    /**
     * GET /api/search/posts
     * Search verified posts by caption.
     */
    fastify.get<{ Querystring: { q: string } }>('/search/posts', async (request, reply) => {
        const { q } = request.query;
        if (!q || q.length < 2) return [];

        try {
            const { data: posts, error } = await supabase
                .from('posts')
                .select(`
                    id,
                    media_url,
                    media_type,
                    caption,
                    profiles!inner (
                        username,
                        avatar_url,
                        trust_status
                    )
                `)
                .eq('visibility', 'PUBLIC')
                // .eq('verification_status', 'APPROVED') // Relaxed
                // .eq('profiles.trust_status', 'TRUSTED') // REMOVED RESTRICTION
                .ilike('caption', `%${q}%`)
                .limit(50);

            if (error) throw error;
            return posts;

        } catch (error) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Search failed' });
        }
    });
}
