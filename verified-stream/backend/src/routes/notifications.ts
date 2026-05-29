import { FastifyInstance } from 'fastify';
import { supabase } from '../supabase.js';

export async function notificationRoutes(fastify: FastifyInstance) {

    /**
     * GET /api/notifications
     * Fetch user notifications
     */
    fastify.get('/', async (request, reply) => {
        const authHeader = request.headers.authorization;
        if (!authHeader) return reply.code(401).send({ error: 'Unauthorized' });

        const { limit = '30', offset = '0' } = request.query as { limit?: string; offset?: string };

        try {
            const token = authHeader.replace('Bearer ', '');
            const { data: { user } } = await supabase.auth.getUser(token);
            if (!user) return reply.code(401).send({ error: 'Invalid token' });

            const { data: notifications, error } = await supabase
                .from('notifications')
                .select(`
                    *,
                    related_user:related_user_id (
                        username, display_name, avatar_url
                    )
                `)
                .eq('user_id', user.id)
                .order('created_at', { ascending: false })
                .range(parseInt(offset), parseInt(offset) + parseInt(limit) - 1);

            if (error) throw error;

            // Count unread
            const { count: unreadCount } = await supabase
                .from('notifications')
                .select('*', { count: 'exact', head: true })
                .eq('user_id', user.id)
                .eq('is_read', false);

            return {
                notifications: notifications || [],
                unreadCount: unreadCount || 0
            };
        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to fetch notifications' });
        }
    });

    /**
     * POST /api/notifications/read
     * Mark notifications as read
     */
    fastify.post('/read', async (request, reply) => {
        const authHeader = request.headers.authorization;
        if (!authHeader) return reply.code(401).send({ error: 'Unauthorized' });

        const { notificationIds } = request.body as { notificationIds?: string[] };

        try {
            const token = authHeader.replace('Bearer ', '');
            const { data: { user } } = await supabase.auth.getUser(token);
            if (!user) return reply.code(401).send({ error: 'Invalid token' });

            let query = supabase
                .from('notifications')
                .update({ is_read: true })
                .eq('user_id', user.id);

            if (notificationIds && notificationIds.length > 0) {
                query = query.in('id', notificationIds);
            }

            const { error } = await query;
            if (error) throw error;

            return { success: true };
        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to mark as read' });
        }
    });

    /**
     * GET /api/notifications/unread-count
     * Get unread notification count
     */
    fastify.get('/unread-count', async (request, reply) => {
        const authHeader = request.headers.authorization;
        if (!authHeader) return reply.code(401).send({ error: 'Unauthorized' });

        try {
            const token = authHeader.replace('Bearer ', '');
            const { data: { user } } = await supabase.auth.getUser(token);
            if (!user) return reply.code(401).send({ error: 'Invalid token' });

            const { count } = await supabase
                .from('notifications')
                .select('*', { count: 'exact', head: true })
                .eq('user_id', user.id)
                .eq('is_read', false);

            return { count: count || 0 };
        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to get count' });
        }
    });

    /**
     * GET /api/notifications/alerts
     * Get active deepfake alerts (public)
     */
    fastify.get('/alerts', async (request, reply) => {
        try {
            const { data: alerts, error } = await supabase
                .from('deepfake_alerts')
                .select('*')
                .eq('is_active', true)
                .order('created_at', { ascending: false })
                .limit(10);

            if (error) throw error;
            return { alerts: alerts || [] };
        } catch (error: any) {
            fastify.log.error(error);
            return reply.code(500).send({ error: 'Failed to fetch alerts' });
        }
    });
}
