import { FastifyInstance } from 'fastify';
import { supabase } from '../supabase.js';

export async function transparencyRoutes(fastify: FastifyInstance) {

    /**
     * GET /api/transparency/post/:postId
     * Content Transparency Panel - Returns full verification details for a post
     */
    fastify.get<{ Params: { postId: string } }>('/post/:postId', async (request, reply) => {
        const { postId } = request.params;

        try {
            // Fetch post with verification data
            const { data: post, error } = await supabase
                .from('posts')
                .select(`
                    id, media_type, caption, created_at, authenticity_label, upload_source,
                    media_hash_check, content_hash_proof,
                    verification:verification_log_id (
                        id, verdict, score, deepfake_verdict, fake_news_verdict,
                        final_verdict, model_score, final_score, model_name, model_version,
                        authenticity_label, score_breakdown, media_type, upload_source,
                        device_metadata, created_at
                    ),
                    profiles:user_id (
                        username, display_name, trust_score, trust_status,
                        is_verified_creator, identity_verified
                    )
                `)
                .eq('id', postId)
                .single();

            if (error || !post) {
                return reply.code(404).send({ error: 'Post not found' });
            }

            const verification = post.verification as any;
            const scoreBreakdown = verification?.score_breakdown || {};

            // Calculate authenticity score (inverse of fake score)
            const authenticityScore = verification
                ? Math.round((1 - (verification.final_score || 0)) * 100)
                : 100;

            const deepfakeProbability = verification
                ? Math.round((verification.model_score || 0) * 100)
                : 0;

            const aiGeneratedProbability = verification
                ? Math.round(((scoreBreakdown.artifact_score || 0)) * 100)
                : 0;

            // Fetch content proof and flags in parallel
            const [proofRes, flagCountRes, confirmedFlagsRes] = await Promise.all([
                supabase
                    .from('content_proofs')
                    .select('*')
                    .eq('post_id', postId)
                    .single(),
                supabase
                    .from('community_flags')
                    .select('*', { count: 'exact', head: true })
                    .eq('post_id', postId),
                supabase
                    .from('community_flags')
                    .select('*', { count: 'exact', head: true })
                    .eq('post_id', postId)
                    .eq('status', 'CONFIRMED')
            ]);

            const proof = proofRes.data;
            const flagCount = flagCountRes.count;
            const confirmedFlags = confirmedFlagsRes.count;

            return {
                postId: post.id,
                authenticityScore,
                deepfakeProbability,
                aiGeneratedProbability,
                metadataIntegrity: verification?.device_metadata ? 'Valid' : 'Unknown',
                uploadSource: post.upload_source || 'GALLERY',
                uploadType: post.upload_source === 'CAMERA' ? 'Camera Capture' : 'Gallery Upload',
                mediaType: post.media_type,
                authenticityLabel: post.authenticity_label || 'VERIFIED_REAL',
                verifiedAt: verification?.created_at,
                modelUsed: verification?.model_name || 'efficientnet-b0',
                modelVersion: verification?.model_version || '1.0',
                scoreBreakdown: {
                    neuralNetwork: scoreBreakdown.model_score || 0,
                    artifactAnalysis: scoreBreakdown.artifact_score || 0,
                    temporalConsistency: scoreBreakdown.temporal_score || 0,
                    metadataScan: scoreBreakdown.metadata_score || 0,
                    compressionAnalysis: scoreBreakdown.compression_score || 0
                },
                author: post.profiles,
                contentProof: proof ? {
                    proofHash: proof.proof_hash,
                    chainIndex: proof.proof_chain_index,
                    verifiedAt: proof.verified_at
                } : null,
                communityFlags: {
                    total: flagCount || 0,
                    confirmed: confirmedFlags || 0
                }
            };
        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to fetch transparency data' });
        }
    });

    /**
     * GET /api/transparency/timeline/:username
     * Authenticity Timeline - Shows verification history summary for a user
     */
    fastify.get<{ Params: { username: string }; Querystring: { days?: string } }>(
        '/timeline/:username',
        async (request, reply) => {
            const { username } = request.params;
            const days = parseInt(request.query.days || '30');

            try {
                const { data: profile } = await supabase
                    .from('profiles')
                    .select('id')
                    .eq('username', username)
                    .single();

                if (!profile) return reply.code(404).send({ error: 'User not found' });

                const since = new Date(Date.now() - days * 86400000).toISOString();

                const { data: logs, error } = await supabase
                    .from('verification_logs')
                    .select('final_verdict, authenticity_label, created_at, media_type')
                    .eq('user_id', profile.id)
                    .gte('created_at', since)
                    .order('created_at', { ascending: false });

                if (error) throw error;

                const totalPosts = logs?.length || 0;
                const authentic = logs?.filter((l: any) =>
                    l.final_verdict === 'REAL' || l.final_verdict === 'APPROVED'
                ).length || 0;
                const edited = logs?.filter((l: any) =>
                    l.authenticity_label === 'EDITED'
                ).length || 0;
                const blocked = logs?.filter((l: any) =>
                    l.final_verdict === 'FAKE' || l.final_verdict === 'REJECTED'
                ).length || 0;

                // Group by day
                const dailyBreakdown: Record<string, { authentic: number; blocked: number }> = {};
                logs?.forEach((log: any) => {
                    const day = new Date(log.created_at).toISOString().split('T')[0];
                    if (!dailyBreakdown[day]) dailyBreakdown[day] = { authentic: 0, blocked: 0 };
                    if (log.final_verdict === 'REAL' || log.final_verdict === 'APPROVED') {
                        dailyBreakdown[day].authentic++;
                    } else {
                        dailyBreakdown[day].blocked++;
                    }
                });

                return {
                    period: `Last ${days} days`,
                    totalPosts,
                    authentic,
                    edited,
                    blocked,
                    authenticityRate: totalPosts > 0 ? Math.round((authentic / totalPosts) * 100) : 100,
                    dailyBreakdown: Object.entries(dailyBreakdown).map(([date, counts]) => ({
                        date,
                        ...counts
                    }))
                };
            } catch (error: any) {
                fastify.log.error(error);
                return reply.code(500).send({ error: 'Failed to fetch timeline' });
            }
        }
    );
}
