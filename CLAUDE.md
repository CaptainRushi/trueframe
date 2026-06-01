# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TrueFrame is an AI-powered deepfake detection and content verification platform. Users upload images/videos, which pass through an AI verification pipeline. Only content verified as "real" gets published to a social feed. The system is **fail-closed**: any error, timeout, or ambiguous result blocks the content.

## Repository Layout

The actual application lives in `verified-stream/`. All paths below are relative to that directory.

- `src/` — React frontend (Vite + SWC)
- `backend/src/` — Fastify API server (Node.js/TypeScript)
- `ai_service/` — Python deepfake detection engine (EfficientNet ONNX, MTCNN, OpenCV)
- `backend/db/` — SQL schema and migrations (run against Supabase)

## Development Commands

### Frontend (from `verified-stream/`)
```bash
npm run dev        # Vite dev server at http://localhost:8080
npm run build      # Production build
npm run lint       # ESLint
```

### Backend (from `verified-stream/backend/`)
```bash
npm run dev        # Fastify with tsx watch at http://localhost:3001
npm run build      # Compile TS to dist/
npm run start      # Run compiled output
```

### AI Service (called automatically by backend via spawn)
```bash
python3 ai_service/main.py <file_path>       # Deepfake detection
python3 ai_service/context_verify.py          # Fake news caption analysis
```

No test runner is currently configured.

## Architecture

### Verification Pipeline (core business logic)

`POST /api/upload/verify-upload` is the critical path:

1. Authenticate via Supabase JWT (Bearer token)
2. Save file to `/tmp/uploads`, compute SHA256 hash
3. Spawn `ai_service/main.py` — deepfake detection with weighted score fusion:
   - EfficientNet ONNX model (0.45) + artifact analysis (0.25) + temporal consistency (0.20) + metadata scan (0.10)
4. Verdict (3-tier): < 0.60 → APPROVED, 0.60–0.74 → UNDER_REVIEW, >= 0.75 → REJECTED
5. If approved, spawn `ai_service/context_verify.py` for caption/fake-news check
6. Log result to `verification_logs`, update user trust score, upload to Supabase Storage if REAL

### Frontend Architecture

- **Router**: React Router v6 in `src/App.tsx` with `ProtectedRoute` wrapper for auth-gated pages
- **UI**: shadcn-ui components in `src/components/ui/`, feature components in `src/components/{feed,upload,auth,layout}/`
- **State**: TanStack React Query for server state, Supabase Auth for session
- **Styling**: Tailwind CSS (dark-mode-first, class strategy), custom animations in `tailwind.config.ts`
- **Path alias**: `@/*` maps to `src/*`

### Backend Architecture

- **Framework**: Fastify with CORS and multipart plugins (50MB upload limit)
- **Routes**: `backend/src/routes/` — upload, feed, dashboard, profile, social, account, explore
- **Database**: Supabase PostgreSQL via `@supabase/supabase-js` (service role on backend, anon key on frontend)
- **Auth**: Supabase Auth (Google/GitHub OAuth), JWT verified on each API call

### Database Tables

Core tables: `profiles` (trust_status, real/fake percentages), `verification_logs` (every upload attempt with scores/verdicts), `posts` (only verified-real content), `likes`, `comments`, `shares`. Schema in `backend/db/schema.sql`.

### Trust System

Users have trust_status: `NEW_USER` → `TRUSTED` → `AT_RISK` → `RESTRICTED` based on their `real_percentage`. Over 30% fake content triggers restriction.

## Environment Variables

**Frontend** (`.env.development`): `VITE_BACKEND_URL`, `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`

**Backend** (`backend/.env`): `PORT`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_ANON_KEY`, `DATABASE_URL`, `REDIS_URL`, `PYTHON_PATH`, `NODE_ENV`, `ALLOWED_ORIGIN`

## Key Conventions

- TypeScript with relaxed strictness (noImplicitAny: false, strictNullChecks: false)
- Zod validation on API inputs
- All UI is dark-mode-first
- Fail-closed verification: Python error, timeout, or no face detected → block upload
- Service role key is backend-only; anon key is frontend-only
- Frontend dev server runs on port 8080, backend on port 3001
