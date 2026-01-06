import { FastifyInstance } from 'fastify';
import { supabase } from '../supabase.js';

interface DashboardStats {
    totalUploads: number;
    verifiedUploads: number;
    rejectedUploads: number;
    realPercentage: number;
    fakePercentage: number;
    fakeDetails: {
        total: number;
        approved: number;
        rejected: number;
        fakeRatio: number;
    };
    status: 'TRUSTED' | 'AT_RISK' | 'RESTRICTED' | 'NEW_USER';
}

function calculateUserStatus(fakePercentage: number, totalUploads: number): DashboardStats['status'] {
    if (totalUploads === 0) return 'NEW_USER';
    if (fakePercentage > 30) return 'RESTRICTED';
    if (fakePercentage > 10) return 'AT_RISK';
    return 'TRUSTED';
}

export async function dashboardRoutes(fastify: FastifyInstance) {

    fastify.get('/stats', async (request, reply) => {
        let { userId } = request.query as { userId?: string };

        // --- AUTH: Extract userId from token if missing ---
        if (!userId) {
            const authHeader = request.headers.authorization;
            if (authHeader) {
                const token = authHeader.replace('Bearer ', '');
                const { data: { user } } = await supabase.auth.getUser(token);
                userId = user?.id;
            }
        }

        if (!userId) {
            return reply.code(400).send({ error: 'Missing userId parameter or invalid token' });
        }

        try {
            // SINGLE SOURCE OF TRUTH: verification_logs
            // Total Attempts
            const { count: total } = await supabase
                .from('verification_logs')
                .select('*', { count: 'exact', head: true })
                .eq('user_id', userId);

            // Real Uploads (Passed Everything)
            const { count: real } = await supabase
                .from('verification_logs')
                .select('*', { count: 'exact', head: true })
                .eq('user_id', userId)
                .in('final_verdict', ['REAL', 'APPROVED']);

            const totalUploads = total || 0;
            const verifiedUploads = real || 0;
            const rejectedUploads = totalUploads - verifiedUploads;

            // PERCENTAGE CALCULATION (Backend Only)
            const realPercentage = totalUploads > 0 ? Math.round((verifiedUploads / totalUploads) * 100) : 100;
            const fakePercentage = totalUploads > 0 ? Math.round((rejectedUploads / totalUploads) * 100) : 0;

            const status = calculateUserStatus(fakePercentage, totalUploads);

            return {
                totalUploads,
                verifiedUploads,
                rejectedUploads,
                realPercentage,
                fakePercentage,
                fakeDetails: {
                    total: totalUploads,
                    approved: verifiedUploads,
                    rejected: rejectedUploads,
                    fakeRatio: fakePercentage
                },
                status
            } as DashboardStats;

        } catch (error) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to fetch stats' });
        }
    });

    fastify.get('/history', async (request, reply) => {
        let { userId, limit = '20' } = request.query as { userId?: string; limit?: string };

        // --- AUTH: Extract userId from token if missing ---
        if (!userId) {
            const authHeader = request.headers.authorization;
            if (authHeader) {
                const token = authHeader.replace('Bearer ', '');
                const { data: { user } } = await supabase.auth.getUser(token);
                userId = user?.id;
            }
        }

        if (!userId) {
            return reply.code(400).send({ error: 'Missing userId parameter or invalid token' });
        }

        try {
            const { data: history, error } = await supabase
                .from('verification_logs')
                .select('*')
                .eq('user_id', userId)
                .order('created_at', { ascending: false })
                .limit(parseInt(limit));

            if (error) throw error;

            return {
                history: history?.map((h: any) => ({
                    id: h.id,
                    created_at: h.created_at,
                    verdict: h.verdict,
                    score: h.score,
                    reason: h.reason,
                    media_type: h.media_type
                }))
            };

        } catch (error) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to fetch history' });
        }
    });
}
