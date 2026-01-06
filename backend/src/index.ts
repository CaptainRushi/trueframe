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

dotenv.config();

const server = fastify({
  logger: true,
  bodyLimit: 50 * 1024 * 1024, // 50MB limit
});

// Register Plugins
server.register(cors, {
  origin: process.env.ALLOWED_ORIGIN || '*',
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
    await server.listen({ port, host: '0.0.0.0' });
    console.log(`Server running at http://localhost:${port}`);
  } catch (err) {
    server.log.error(err);
    process.exit(1);
  }
};

start();
