import { FastifyInstance } from 'fastify';
import { supabase } from '../supabase.js';

export async function accountRoutes(fastify: FastifyInstance) {

    /**
     * POST /api/account/signout
     * Securely invalidates session
     */
    fastify.post('/signout', async (request, reply) => {
        const authHeader = request.headers.authorization;
        if (!authHeader) return reply.code(400).send({ error: 'No session to sign out' });

        try {
            const token = authHeader.replace('Bearer ', '');
            // Invalidate session server-side
            const { error } = await supabase.auth.admin.signOut(token);
            if (error) {
                // Even if error (token expired), we return success to clear frontend
                console.warn('Signout warning:', error.message);
            }
            return { success: true, message: 'Signed out successfully' };
        } catch (error) {
            fastify.log.error(error);
            return { success: true }; // Always return success to allow frontend cleanup
        }
    });

    /**
     * DELETE /api/account
     * Irreversible Account Deletion - Full Data Purge
     */
    fastify.delete('/', async (request, reply) => {
        const authHeader = request.headers.authorization;
        if (!authHeader) return reply.code(401).send({ error: 'Unauthorized' });

        try {
            // 1. Validate Auth
            const token = authHeader.replace('Bearer ', '');
            const { data: { user }, error: authError } = await supabase.auth.getUser(token);

            if (authError || !user) {
                return reply.code(401).send({ error: 'Invalid authentication' });
            }

            const userId = user.id;

            // 2. Fetch all user data to clean up storage
            const { data: profile } = await supabase
                .from('profiles')
                .select('avatar_url')
                .eq('id', userId)
                .single();

            const { data: userPosts } = await supabase
                .from('posts')
                .select('media_url')
                .eq('user_id', userId);

            // 3. Clean up Storage (Avatars)
            if (profile?.avatar_url) {
                try {
                    const urlParts = profile.avatar_url.split('/storage/v1/object/public/avatars/');
                    if (urlParts.length > 1) {
                        const storagePath = urlParts[1];
                        await supabase.storage.from('avatars').remove([storagePath]);
                    }
                } catch (e) {
                    console.error('Avatar storage deletion failed', e);
                }
            }

            // 4. Clean up Storage (Post Media)
            if (userPosts && userPosts.length > 0) {
                const postStoragePaths = userPosts
                    .map((p: { media_url: string }) => {
                        const parts = p.media_url.split('/storage/v1/object/public/posts/');
                        return parts.length > 1 ? parts[1] : null;
                    })
                    .filter((path: string | null): path is string => path !== null);

                if (postStoragePaths.length > 0) {
                    try {
                        await supabase.storage.from('posts').remove(postStoragePaths);
                    } catch (e) {
                        console.error('Post media storage deletion failed', e);
                    }
                }
            }

            // 5. Delete Database Records (Order matters for FKs)

            // Step A: Delete social interactions performed BY this user
            await supabase.from('likes').delete().eq('user_id', userId);
            await supabase.from('comments').delete().eq('user_id', userId);
            await supabase.from('shares').delete().eq('user_id', userId);

            // Step B: Delete social interactions ON the user's posts
            const { data: userPostsList } = await supabase.from('posts').select('id').eq('user_id', userId);
            if (userPostsList && userPostsList.length > 0) {
                const postIds = userPostsList.map((p: { id: string }) => p.id);
                await supabase.from('likes').delete().in('post_id', postIds);
                await supabase.from('comments').delete().in('post_id', postIds);
                await supabase.from('shares').delete().in('post_id', postIds);
            }

            // Step C: Delete Follows
            await supabase.from('follows').delete().or(`follower_id.eq.${userId},following_id.eq.${userId}`);

            // Delete Posts (This handles cascading for social interactions ON these posts)
            // Posts must be deleted before verification_logs because posts reference them
            await supabase.from('posts').delete().eq('user_id', userId);

            // Delete Verification Logs
            await supabase.from('verification_logs').delete().eq('user_id', userId);

            // Delete Profile
            const { error: profileError } = await supabase
                .from('profiles')
                .delete()
                .eq('id', userId);

            if (profileError) throw profileError;

            // 6. Delete Auth User (Prevents login)
            const { error: deleteError } = await supabase.auth.admin.deleteUser(userId);

            if (deleteError) {
                console.error('Auth user deletion failed:', deleteError);
                throw deleteError;
            }

            // 7. Success
            return { success: true, message: 'Account and all associated data permanently deleted' };

        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Account deletion failed: ' + error.message });
        }
    });
}
