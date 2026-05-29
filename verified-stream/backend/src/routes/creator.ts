import { FastifyInstance } from 'fastify';
import { supabase } from '../supabase.js';

export async function creatorRoutes(fastify: FastifyInstance) {

    /**
     * GET /api/creator/status
     * Check if user qualifies for Verified Creator status
     */
    fastify.get('/status', async (request, reply) => {
        const authHeader = request.headers.authorization;
        if (!authHeader) return reply.code(401).send({ error: 'Unauthorized' });

        try {
            const token = authHeader.replace('Bearer ', '');
            const { data: { user } } = await supabase.auth.getUser(token);
            if (!user) return reply.code(401).send({ error: 'Invalid token' });

            const { data: profile } = await supabase
                .from('profiles')
                .select('*')
                .eq('id', user.id)
                .single();

            if (!profile) return reply.code(404).send({ error: 'Profile not found' });

            const trustScore = profile.trust_score || 0;
            const totalAttempts = profile.total_attempts || 0;
            const realPercentage = profile.real_percentage || 0;

            // Creator qualification criteria
            const criteria = {
                trustScore: { required: 85, current: trustScore, met: trustScore >= 85 },
                totalUploads: { required: 10, current: totalAttempts, met: totalAttempts >= 10 },
                authenticityRate: { required: 90, current: realPercentage, met: realPercentage >= 90 },
                identityVerified: { required: true, current: profile.identity_verified || false, met: profile.identity_verified || false }
            };

            const allCriteriaMet = criteria.trustScore.met &&
                criteria.totalUploads.met &&
                criteria.authenticityRate.met;

            return {
                isVerifiedCreator: profile.is_verified_creator || false,
                verifiedAt: profile.verified_creator_at,
                eligible: allCriteriaMet,
                criteria,
                benefits: [
                    'Verified Authentic Creator badge',
                    'Higher visibility in feed ranking',
                    'Priority content verification',
                    'Community Verifier eligibility',
                    'Authenticity badge on all posts'
                ]
            };
        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to check creator status' });
        }
    });

    /**
     * POST /api/creator/apply
     * Apply for Verified Creator program
     */
    fastify.post('/apply', async (request, reply) => {
        const authHeader = request.headers.authorization;
        if (!authHeader) return reply.code(401).send({ error: 'Unauthorized' });

        try {
            const token = authHeader.replace('Bearer ', '');
            const { data: { user } } = await supabase.auth.getUser(token);
            if (!user) return reply.code(401).send({ error: 'Invalid token' });

            const { data: profile } = await supabase
                .from('profiles')
                .select('trust_score, total_attempts, real_percentage, is_verified_creator')
                .eq('id', user.id)
                .single();

            if (!profile) return reply.code(404).send({ error: 'Profile not found' });

            if (profile.is_verified_creator) {
                return reply.code(400).send({ error: 'Already a Verified Creator' });
            }

            const trustScore = profile.trust_score || 0;
            const totalAttempts = profile.total_attempts || 0;
            const realPercentage = profile.real_percentage || 0;

            if (trustScore < 85 || totalAttempts < 10 || realPercentage < 90) {
                return reply.code(400).send({
                    error: 'Eligibility criteria not met',
                    requirements: {
                        trustScore: { required: 85, current: trustScore },
                        totalUploads: { required: 10, current: totalAttempts },
                        authenticityRate: { required: 90, current: realPercentage }
                    }
                });
            }

            // Grant Verified Creator status
            const { error } = await supabase
                .from('profiles')
                .update({
                    is_verified_creator: true,
                    verified_creator_at: new Date().toISOString(),
                    is_community_verifier: true
                })
                .eq('id', user.id);

            if (error) throw error;

            // Send notification
            await supabase.from('notifications').insert({
                user_id: user.id,
                type: 'CREATOR_BADGE',
                title: 'Verified Authentic Creator',
                message: 'Congratulations! You have been awarded the Verified Authentic Creator badge for your outstanding authenticity record.'
            });

            return { success: true, message: 'You are now a Verified Authentic Creator!' };
        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to apply for creator program' });
        }
    });

    /**
     * GET /api/creator/leaderboard
     * Top verified creators
     */
    fastify.get('/leaderboard', async (request, reply) => {
        try {
            const { data: creators, error } = await supabase
                .from('profiles')
                .select('id, username, display_name, avatar_url, trust_score, real_percentage, total_attempts, is_verified_creator, identity_verified')
                .gte('trust_score', 50)
                .gte('total_attempts', 1)
                .order('trust_score', { ascending: false })
                .limit(25);

            if (error) throw error;

            const leaderboard = (creators || []).map((c: any, index: number) => ({
                rank: index + 1,
                ...c,
                trustLevel: getTrustLevel(c.trust_score)
            }));

            return { leaderboard };
        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to fetch leaderboard' });
        }
    });
}

function getTrustLevel(score: number): string {
    if (score >= 90) return 'Trusted Creator';
    if (score >= 70) return 'Reliable User';
    if (score >= 50) return 'Normal User';
    if (score >= 30) return 'Suspicious';
    return 'High Risk';
}
