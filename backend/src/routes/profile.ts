import { FastifyInstance, FastifyRequest } from 'fastify';
import '@fastify/multipart';
import { supabase } from '../supabase.js';

interface ProfileParams {
    username: string;
}

const AVATAR_SIZE_LIMIT = 300 * 1024; // 300KB
const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

export async function profileRoutes(fastify: FastifyInstance) {
    // Ensure storage buckets exist
    const ensureBuckets = async () => {
        try {
            const { data: buckets } = await supabase.storage.listBuckets();
            const bucketNames = buckets?.map((b: any) => b.name) || [];
            if (!bucketNames.includes('avatars')) {
                await supabase.storage.createBucket('avatars', { public: true });
            }
        } catch (e) {
            console.warn('Storage bucket check/creation failed - check permissions');
        }
    };
    ensureBuckets();

    /**
     * POST /api/profile/avatar
     * Upload profile avatar (Max 300KB)
     */
    fastify.post('/avatar', async (request, reply) => {
        const authHeader = request.headers.authorization;
        if (!authHeader) {
            return reply.code(401).send({ error: 'Unauthorized' });
        }

        try {
            const data = await request.file();
            if (!data) {
                return reply.code(400).send({ error: 'No file uploaded' });
            }

            // Validate Size (Check buffer length if needed, or rely on busboy limits if configured, 
            // but for strict logic we read the file)
            const buffer = await data.toBuffer();

            if (buffer.length > AVATAR_SIZE_LIMIT) {
                return reply.code(400).send({ error: 'Avatar must be less than 300KB' });
            }

            // Validate Type
            if (!ALLOWED_MIME_TYPES.includes(data.mimetype)) {
                return reply.code(400).send({ error: 'Invalid file type. Allowed: JPG, PNG, WEBP' });
            }

            const token = authHeader.replace('Bearer ', '');
            const { data: { user }, error: authError } = await supabase.auth.getUser(token);

            if (authError || !user) {
                return reply.code(401).send({ error: 'Invalid token' });
            }

            // Upload to Supabase Storage
            const fileName = `${user.id}/${Date.now()}_avatar.${data.mimetype.split('/')[1]}`;
            const { data: uploadData, error: uploadError } = await supabase.storage
                .from('avatars')
                .upload(fileName, buffer, {
                    contentType: data.mimetype,
                    upsert: true
                });

            if (uploadError) {
                throw uploadError;
            }

            // Get Public URL
            const { data: { publicUrl } } = supabase.storage
                .from('avatars')
                .getPublicUrl(fileName);

            // Update Profile
            await supabase
                .from('profiles')
                .update({ avatar_url: publicUrl })
                .eq('id', user.id);

            return { avatarUrl: publicUrl };

        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Avatar upload failed' });
        }
    });

    /**
     * POST /api/profile/init
     * Initialize profile for authenticated user
     */
    fastify.post('/init', async (request, reply) => {
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

            // Check if profile exists
            const { data: existingProfile } = await supabase
                .from('profiles')
                .select('*')
                .eq('id', user.id)
                .single();

            if (existingProfile) {
                return existingProfile;
            }

            // Create new profile
            // Use metadata name or email part as username
            let username = user.user_metadata?.name?.replace(/\s+/g, '').toLowerCase() ||
                user.email?.split('@')[0] ||
                `user_${user.id.slice(0, 8)}`;

            // Ensure uniqueness (simple append)
            const { data: profile, error: createError } = await supabase
                .from('profiles')
                .insert({
                    id: user.id,
                    username: username,
                    display_name: user.user_metadata?.name || username,
                    avatar_url: user.user_metadata?.avatar_url
                })
                .select()
                .single();

            if (createError) {
                // If conflict, try appending random string
                username = `${username}_${Math.floor(Math.random() * 1000)}`;
                const { data: profileRetry, error: retryError } = await supabase
                    .from('profiles')
                    .insert({
                        id: user.id,
                        username: username,
                        display_name: user.user_metadata?.name || username,
                        avatar_url: user.user_metadata?.avatar_url
                    })
                    .select()
                    .single();

                if (retryError) throw retryError;
                return profileRetry;
            }

            return profile;

        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Profile init failed' });
        }
    });

    /**
     * GET /api/profile/me/history
     * Private verification history
     */
    fastify.get('/me/history', async (request, reply) => {
        const authHeader = request.headers.authorization;
        if (!authHeader) {
            return reply.code(401).send({ error: 'Unauthorized' });
        }

        const token = authHeader.replace('Bearer ', '');
        const { data: { user }, error: authError } = await supabase.auth.getUser(token);

        if (authError || !user) {
            return reply.code(401).send({ error: 'Invalid token' });
        }

        try {
            const { data: history, error } = await supabase
                .from('verification_logs')
                .select('*')
                .eq('user_id', user.id)
                .order('created_at', { ascending: false });

            if (error) throw error;

            return history;
        } catch (error) {
            return reply.code(500).send({ error: 'Failed to fetch history' });
        }
    });

    /**
     * GET /api/profile/:username
     * Public profile data
     */
    fastify.get<{ Params: ProfileParams }>('/:username', async (request, reply) => {
        const { username } = request.params;

        try {
            // 1. Get Profile
            const { data: profile, error: profileError } = await supabase
                .from('profiles')
                .select('*')
                .eq('username', username)
                .single();

            if (profileError || !profile) {
                return reply.code(404).send({ error: 'User not found' });
            }

            const userId = profile.id;

            // Prepare independent queries
            const authHeader = request.headers.authorization;

            const fetchFollowStatus = async () => {
                if (!authHeader) return false;
                try {
                    const token = authHeader.replace('Bearer ', '');
                    const { data: { user } } = await supabase.auth.getUser(token);
                    if (user && user.id !== userId) {
                        const { data: follow } = await supabase
                            .from('follows')
                            .select('id')
                            .eq('follower_id', user.id)
                            .eq('following_id', userId)
                            .single();
                        return !!follow;
                    }
                } catch (e) {
                    return false;
                }
                return false;
            };

            // SINGLE SOURCE OF TRUTH: verification_logs
            const fetchTotalAttempts = supabase
                .from('verification_logs')
                .select('*', { count: 'exact', head: true })
                .eq('user_id', userId);

            const fetchRealUploads = supabase
                .from('verification_logs')
                .select('*', { count: 'exact', head: true })
                .eq('user_id', userId)
                .in('final_verdict', ['REAL', 'APPROVED']);

            // Limit posts to 50 for performance
            const fetchPosts = supabase
                .from('posts')
                .select('*')
                .eq('user_id', userId)
                .order('created_at', { ascending: false })
                .limit(50);

            // Execute in parallel
            const [
                isFollowing,
                { count: total },
                { count: real },
                { data: posts }
            ] = await Promise.all([
                fetchFollowStatus(),
                fetchTotalAttempts,
                fetchRealUploads,
                fetchPosts
            ]);

            // 3. Calculate Stats
            const totalUploads = total || 0;
            const verifiedUploads = real || 0;
            const blockedUploads = totalUploads - verifiedUploads;

            const realPercentage = totalUploads > 0 ? Math.round((verifiedUploads / totalUploads) * 100) : 100;
            const fakePercentage = totalUploads > 0 ? Math.round((blockedUploads / totalUploads) * 100) : 0;

            let status = 'TRUSTED';
            if (totalUploads === 0) {
                status = 'NEW_USER';
            } else if (fakePercentage > 30) {
                status = 'RESTRICTED';
            } else if (fakePercentage > 10) {
                status = 'AT_RISK';
            }

            // 4. Organize Content
            const imagePosts = posts?.filter((p: any) => p.media_type === 'image') || [];
            const videoReels = posts?.filter((p: any) => p.media_type === 'video') || [];

            return {
                id: profile.id, // Expose ID for ownership check
                username: profile.username,
                displayName: profile.display_name,
                avatarUrl: profile.avatar_url,
                bio: profile.bio,
                followersCount: profile.followers_count || 0,
                followingCount: profile.following_count || 0,
                isFollowing,
                totalUploads,
                realUploads: verifiedUploads,
                fakeUploads: blockedUploads,
                realPercentage,
                fakePercentage,
                fake_percentage: profile.fake_percentage ?? fakePercentage,
                total_attempts: profile.total_attempts ?? totalUploads,
                real_count: profile.real_count ?? verifiedUploads,
                fake_count: profile.fake_count ?? blockedUploads,
                status,
                posts: imagePosts,
                reels: videoReels
            };

        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to fetch profile' });
        }
    });


    /**
     * POST /api/profile/update
     * Update profile details (display_name, etc.)
     */
    fastify.post('/update', async (request, reply) => {
        const authHeader = request.headers.authorization;
        if (!authHeader) {
            return reply.code(401).send({ error: 'Unauthorized' });
        }

        const { displayName, bio } = request.body as { displayName?: string; bio?: string };

        try {
            const token = authHeader.replace('Bearer ', '');
            const { data: { user }, error: authError } = await supabase.auth.getUser(token);

            if (authError || !user) {
                return reply.code(401).send({ error: 'Invalid token' });
            }

            const { data: updatedProfile, error: updateError } = await supabase
                .from('profiles')
                .update({
                    display_name: displayName,
                    bio: bio
                })
                .eq('id', user.id)
                .select()
                .single();

            if (updateError) throw updateError;

            return updatedProfile;

        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to update profile' });
        }
    });

    /**
     * POST /api/profile/:username/follow
     * Follow a user
     */
    fastify.post<{ Params: ProfileParams }>('/:username/follow', async (request, reply) => {
        const { username } = request.params;
        const authHeader = request.headers.authorization;
        if (!authHeader) return reply.code(401).send({ error: 'Unauthorized' });

        try {
            const token = authHeader.replace('Bearer ', '');
            const { data: { user }, error: authError } = await supabase.auth.getUser(token);
            if (authError || !user) return reply.code(401).send({ error: 'Invalid token' });

            // Get target user ID
            const { data: targetProfile } = await supabase
                .from('profiles')
                .select('id')
                .eq('username', username)
                .single();

            if (!targetProfile) return reply.code(404).send({ error: 'User not found' });
            if (targetProfile.id === user.id) return reply.code(400).send({ error: 'Cannot follow yourself' });

            // Insert follow
            const { error } = await supabase
                .from('follows')
                .insert({ follower_id: user.id, following_id: targetProfile.id });

            if (error) {
                if (error.code === '23505') return { success: true }; // Already following
                throw error;
            }

            return { success: true };
        } catch (error) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to follow user' });
        }
    });

    /**
     * DELETE /api/profile/:username/follow
     * Unfollow a user
     */
    fastify.delete<{ Params: ProfileParams }>('/:username/follow', async (request, reply) => {
        const { username } = request.params;
        const authHeader = request.headers.authorization;
        if (!authHeader) return reply.code(401).send({ error: 'Unauthorized' });

        try {
            const token = authHeader.replace('Bearer ', '');
            const { data: { user }, error: authError } = await supabase.auth.getUser(token);
            if (authError || !user) return reply.code(401).send({ error: 'Invalid token' });

            const { data: targetProfile } = await supabase
                .from('profiles')
                .select('id')
                .eq('username', username)
                .single();

            if (!targetProfile) return reply.code(404).send({ error: 'User not found' });

            const { error } = await supabase
                .from('follows')
                .delete()
                .eq('follower_id', user.id)
                .eq('following_id', targetProfile.id);

            if (error) throw error;

            return { success: true };
        } catch (error) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to unfollow user' });
        }
    });
}
