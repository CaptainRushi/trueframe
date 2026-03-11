import fastify from 'fastify';
import cors from '@fastify/cors';
import multipart from '@fastify/multipart';
import dotenv from 'dotenv';
import { uploadRoutes } from './routes/upload.js';
import { feedRoutes } from './routes/feed.js';
import { dashboardRoutes } from './routes/dashboard.js';
import { profileRoutes } from './routes/profile.js';
import { socialRoutes } from './routes/social.js';
import { accountRoutes } from './routes/account.js';
import { exploreRoutes } from './routes/explore.js';
import { transparencyRoutes } from './routes/transparency.js';
import { communityRoutes } from './routes/community.js';
import { notificationRoutes } from './routes/notifications.js';
import { creatorRoutes } from './routes/creator.js';
import { verificationRoutes } from './routes/verification.js';
import { moderationRoutes } from './routes/moderation.js';

dotenv.config();

const server = fastify({
  logger: true,
  bodyLimit: 50 * 1024 * 1024, // 50MB limit
});

// Register Plugins
const origins = process.env.ALLOWED_ORIGIN
  ? process.env.ALLOWED_ORIGIN.split(',').map(o => o.trim().replace(/\/$/, ''))
  : ['*'];

server.register(cors, {
  origin: origins.length === 1 && origins[0] === '*' ? '*' : origins,
  allowedHeaders: ['Content-Type', 'Authorization'],
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
});

server.register(multipart, {
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB
  }
});

// Register Routes
// Note: Auth is handled by Supabase Auth, no custom auth routes needed
server.register(uploadRoutes, { prefix: '/api/upload' });
server.register(feedRoutes, { prefix: '/api/feed' });
server.register(dashboardRoutes, { prefix: '/api/dashboard' });
server.register(profileRoutes, { prefix: '/api/profile' });
server.register(socialRoutes, { prefix: '/api/social' });
server.register(exploreRoutes, { prefix: '/api' });
server.register(accountRoutes, { prefix: '/api/account' });
server.register(transparencyRoutes, { prefix: '/api/transparency' });
server.register(communityRoutes, { prefix: '/api/community' });
server.register(notificationRoutes, { prefix: '/api/notifications' });
server.register(creatorRoutes, { prefix: '/api/creator' });
server.register(verificationRoutes, { prefix: '/api/verification' });
server.register(moderationRoutes, { prefix: '/api/moderation' });

// Health Check & Root
server.get('/', async () => {
  return {
    message: 'TrueFrame API is live',
    version: '1.0.0',
    stability: 'stable',
    endpoints: ['/api/upload', '/api/feed', '/api/dashboard', '/api/profile', '/api/social']
  };
});

server.get('/health', async (request, reply) => {
  return { status: 'ok', service: 'verified-stream-backend' };
});

const start = async () => {
  try {
    const port = process.env.PORT ? parseInt(process.env.PORT) : 3001;
    const address = await server.listen({ port, host: '::' });
    console.log(`\x1b[32m[SUCCESS]\x1b[0m Backend is live at: ${address}`);
    console.log(`\x1b[36m[INFO]\x1b[0m Local access: http://localhost:${port}`);
  } catch (err) {
    server.log.error(err);
    process.exit(1);
  }
};

start();
