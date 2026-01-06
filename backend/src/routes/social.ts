import { FastifyInstance } from 'fastify';
import { supabase } from '../supabase.js';

export async function socialRoutes(fastify: FastifyInstance) {

    // Toggle Like on a Post
    fastify.post('/like/:postId', async (request, reply) => {
        const { postId } = request.params as { postId: string };
        const authHeader = request.headers.authorization;

        if (!authHeader) {
            return reply.code(401).send({ error: 'Unauthorized' });
        }

        try {
            const token = authHeader.replace('Bearer ', '');
            const { data: { user }, error: authError } = await supabase.auth.getUser(token);

            if (authError || !user) {
                return reply.code(401).send({ error: 'Invalid token' });
            }

            // Check if already liked
            const { data: existingLike } = await supabase
                .from('likes')
                .select('id')
                .eq('post_id', postId)
                .eq('user_id', user.id)
                .single();

            if (existingLike) {
                // Unlike: Delete the like
                await supabase
                    .from('likes')
                    .delete()
                    .eq('id', existingLike.id);

                // Decrement like_count
                await supabase.rpc('decrement_like_count', { post_id: postId });

                return { liked: false, message: 'Post unliked' };
            } else {
                // Like: Create new like
                await supabase
                    .from('likes')
                    .insert({ post_id: postId, user_id: user.id });

                // Increment like_count
                await supabase.rpc('increment_like_count', { post_id: postId });

                return { liked: true, message: 'Post liked' };
            }
        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to toggle like' });
        }
    });

    // Get Like Status for a Post
    fastify.get('/like/:postId/status', async (request, reply) => {
        const { postId } = request.params as { postId: string };
        const authHeader = request.headers.authorization;

        if (!authHeader) {
            return { liked: false };
        }

        try {
            const token = authHeader.replace('Bearer ', '');
            const { data: { user } } = await supabase.auth.getUser(token);

            if (!user) {
                return { liked: false };
            }

            const { data: like } = await supabase
                .from('likes')
                .select('id')
                .eq('post_id', postId)
                .eq('user_id', user.id)
                .single();

            return { liked: !!like };
        } catch (error) {
            return { liked: false };
        }
    });

    // Add Comment to a Post (Trust-Aware)
    fastify.post('/comment/:postId', async (request, reply) => {
        const { postId } = request.params as { postId: string };
        const { content } = request.body as { content: string };
        const authHeader = request.headers.authorization;

        if (!authHeader) return reply.code(401).send({ error: 'Unauthorized' });
        if (!content || content.trim().length === 0) return reply.code(400).send({ error: 'Comment content required' });

        try {
            const token = authHeader.replace('Bearer ', '');
            const { data: { user }, error: authError } = await supabase.auth.getUser(token);

            if (authError || !user) return reply.code(401).send({ error: 'Invalid token' });

            // 1. Fetch User Trust Status
            const { data: profile } = await supabase
                .from('profiles')
                .select('trust_status')
                .eq('id', user.id)
                .single();
            const trustStatus = profile?.trust_status || 'TRUSTED';

            // 2. Classify Comment (CPU-Light Logic)
            const lower = content.toLowerCase();
            let type = 'NORMAL';

            if (/\b(false|fake|wrong|incorrect|lie|debunk|misleading|myth)\b/i.test(lower)) {
                type = 'CORRECTION';
            } else if (content.includes('?') || /\b(why|how|what|who|where|when|can you)\b/i.test(lower)) {
                type = 'QUESTION';
            } else if (/\b(\d+|percent|million|billion|confirmed|breaking|report|source|proven)\b/i.test(lower)) {
                type = 'CLAIM';
            }

            // 3. Apply Trust & Visibility Rules
            let visibility = 'VISIBLE';

            if (type === 'CLAIM') {
                // "Claim comments from low-trust users -> COLLAPSED"
                if (trustStatus !== 'TRUSTED') {
                    visibility = 'COLLAPSED';
                }

                // "High fake probability -> Block" (Simple spam heuristic)
                if (/\b(click|link|buy|crypto|invest|prize)\b/i.test(lower)) {
                    return reply.code(400).send({
                        success: false,
                        reason: "Unverified promotion/spam detected in comment"
                    });
                }
            }

            // 4. Insert Comment
            const { data: comment, error: commentError } = await supabase
                .from('comments')
                .insert({
                    post_id: postId,
                    user_id: user.id,
                    content: content.trim(),
                    type: type,
                    visibility: visibility
                })
                .select()
                .single();

            if (commentError) throw commentError;

            // Increment comment_count
            await supabase.rpc('increment_comment_count', { post_id: postId });

            return { success: true, comment, classification: type, visibility };
        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to add comment' });
        }
    });

    // Get Comments for a Post
    fastify.get('/comments/:postId', async (request, reply) => {
        const { postId } = request.params as { postId: string };

        try {
            const { data: comments, error } = await supabase
                .from('comments')
                .select(`
                    *,
                    profiles:user_id (
                        username,
                        display_name,
                        avatar_url,
                        trust_status
                    )
                `)
                .eq('post_id', postId)
                .order('created_at', { ascending: false });

            if (error) throw error;

            return { comments: comments || [] };
        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to fetch comments' });
        }
    });

    // Track Share
    fastify.post('/share/:postId', async (request, reply) => {
        const { postId } = request.params as { postId: string };
        const authHeader = request.headers.authorization;

        if (!authHeader) {
            return reply.code(401).send({ error: 'Unauthorized' });
        }

        try {
            const token = authHeader.replace('Bearer ', '');
            const { data: { user }, error: authError } = await supabase.auth.getUser(token);

            if (authError || !user) {
                return reply.code(401).send({ error: 'Invalid token' });
            }

            // Track share
            await supabase
                .from('shares')
                .insert({ post_id: postId, user_id: user.id });

            return { success: true, message: 'Share tracked' };
        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to track share' });
        }
    });

    // Delete Post
    fastify.delete('/post/:postId', async (request, reply) => {
        const { postId } = request.params as { postId: string };
        const authHeader = request.headers.authorization;

        if (!authHeader) {
            return reply.code(401).send({ error: 'Unauthorized' });
        }

        try {
            const token = authHeader.replace('Bearer ', '');
            const { data: { user }, error: authError } = await supabase.auth.getUser(token);

            if (authError || !user) {
                return reply.code(401).send({ error: 'Invalid token' });
            }

            // Get post to verify ownership and get media URL
            const { data: post, error: postError } = await supabase
                .from('posts')
                .select('user_id, media_url')
                .eq('id', postId)
                .single();

            if (postError || !post) {
                return reply.code(404).send({ error: 'Post not found' });
            }

            // Verify ownership
            if (post.user_id !== user.id) {
                return reply.code(403).send({ error: 'You can only delete your own posts' });
            }

            // 1. Delete Interactions first to be safe (though CASCADE should handle it)
            await supabase.from('likes').delete().eq('post_id', postId);
            await supabase.from('comments').delete().eq('post_id', postId);
            await supabase.from('shares').delete().eq('post_id', postId);

            // 2. Delete from storage (extract path from URL)
            try {
                const urlParts = post.media_url.split('/storage/v1/object/public/posts/');
                if (urlParts.length > 1) {
                    const storagePath = urlParts[1];
                    await supabase.storage.from('posts').remove([storagePath]);
                }
            } catch (storageError) {
                console.error('Failed to delete from storage:', storageError);
            }

            // 3. Delete post (Finally)
            const { error: deleteError } = await supabase
                .from('posts')
                .delete()
                .eq('id', postId);

            if (deleteError) throw deleteError;

            // 4. Verify deletion (Optional but good for robustness given user's report)
            const { count } = await supabase.from('posts').select('*', { count: 'exact', head: true }).eq('id', postId);
            if (count && count > 0) {
                throw new Error('Post record still exists after deletion attempt');
            }

            return { success: true, message: 'Post deleted successfully' };
        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to delete post' });
        }
    });
}
