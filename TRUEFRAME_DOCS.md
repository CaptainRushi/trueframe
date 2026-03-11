# 🎯 TrueFrame — AI-Powered Deepfake Detection Platform

> **"Only truth gets published."**
> TrueFrame is a social media platform where every piece of content is verified by AI before it reaches the feed. Deepfakes, manipulated images, and fake news are blocked at the gate — and the community helps catch what the AI misses.

---

## 📋 Table of Contents

- [What is TrueFrame?](#what-is-trueframe)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Core Pipelines](#core-pipelines)
  - [Primary Verification Pipeline](#1-primary-verification-pipeline)
  - [Community Reporting Pipeline](#2-community-reporting--secondary-review-pipeline)
  - [Trust Score System](#3-trust-score-system)
- [Database Schema](#database-schema)
- [AI Detection Models](#ai-detection-models)
- [API Reference](#api-reference)
- [Environment Setup](#environment-setup)

---

## 🔍 What is TrueFrame?

TrueFrame is a **fail-closed content verification platform**. Every image or video uploaded by a user passes through a multi-stage AI pipeline before it ever appears on the public feed.

### Core Principles

| Principle | Description |
|-----------|-------------|
| 🔒 **Fail-Closed** | Any error, timeout, or ambiguous result blocks the content |
| 🤖 **AI-First** | Primary + secondary AI models verify every upload |
| 👥 **Community Safety Net** | Users report suspicious posts → triggers secondary AI review |
| 🏆 **Trust-Weighted** | User reputation affects content ranking and moderation influence |
| 🔍 **Transparent** | Every post shows its verification details publicly |

---

## 🛠️ Tech Stack

### 🖥️ Frontend
```
React 18          — UI framework
Vite + SWC        — Build tool (ultra-fast HMR)
TypeScript 5      — Type safety
Tailwind CSS      — Utility-first styling (dark-mode-first)
Framer Motion     — Animations
TanStack Query    — Server state management
React Router v6   — Client-side routing
shadcn/ui         — Base UI component library
Supabase JS       — Auth + realtime
```

### ⚙️ Backend
```
Fastify           — High-performance Node.js HTTP server
TypeScript 5      — Type safety
Supabase          — PostgreSQL database + Auth + Storage
@supabase/supabase-js  — Database client (service role)
@fastify/multipart     — File upload handling (50MB limit)
@fastify/cors          — Cross-origin resource sharing
```

### 🤖 AI Service
```
Python 3          — AI service runtime
ONNX Runtime      — EfficientNet-B0 inference (CPU-optimized)
HuggingFace       — dima806/deepfake_vs_real_image_detection
OpenCV (cv2)      — Frame extraction, face detection, image processing
MTCNN             — Face detection & alignment
NumPy             — Numerical computations
PyExifTool        — Metadata analysis
```

### 🗄️ Database & Storage
```
Supabase PostgreSQL  — Primary database
Supabase Storage     — Media files (posts bucket, public access)
Supabase Auth        — Google / GitHub OAuth + JWT
```

### 🚀 Deployment
```
Vercel            — Frontend hosting
Supabase Cloud    — Database + Auth + Storage
Node.js Server    — Backend API (port 3001)
Python Subprocess — AI service (spawned per request)
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRUEFRAME PLATFORM                           │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                    FRONTEND (React + Vite)                    │  │
│   │                                                              │  │
│   │   Landing  →  Login  →  Feed  →  Upload  →  Dashboard       │  │
│   │                          ↕          ↕                        │  │
│   │             PostCard  FlagModal  UploadStep                  │  │
│   │             TransparencyPanel  ModerationQueue               │  │
│   └──────────────────────┬───────────────┬───────────────────────┘  │
│                           │ HTTP Fetch   │ Supabase Client           │
│                           ↓              ↓                           │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                  BACKEND (Fastify + Node.js)                  │  │
│   │                                                              │  │
│   │  /api/upload     /api/feed      /api/community              │  │
│   │  /api/social     /api/dashboard /api/moderation             │  │
│   │  /api/transparency /api/profile /api/notifications          │  │
│   │                                                              │  │
│   │              ↓ spawn subprocess                              │  │
│   └──────────────────────┬───────────────────────────────────────┘  │
│                           │                                          │
│   ┌───────────────────────┴──────────────────────────────────────┐  │
│   │                   AI SERVICE (Python)                         │  │
│   │                                                              │  │
│   │  main.py              secondary_review.py   context_verify.py│  │
│   │  (Primary Detection)  (Secondary Review)    (Fake News)      │  │
│   │                                                              │  │
│   │  EfficientNet-B0 ONNX + HuggingFace Ensemble                │  │
│   │  Frequency Analysis + GAN Artifact Detection                 │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                           │                                          │
│   ┌───────────────────────┴──────────────────────────────────────┐  │
│   │              SUPABASE (PostgreSQL + Storage + Auth)           │  │
│   │                                                              │  │
│   │  profiles  posts  verification_logs  community_flags         │  │
│   │  secondary_reviews  notifications  trust_score_history       │  │
│   │  likes  comments  shares  follows  content_proofs            │  │
│   └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Core Pipelines

### 1️⃣ Primary Verification Pipeline

Every upload goes through this 10-step pipeline before it can appear on the feed.

```
User Uploads Media
        │
        ▼
┌───────────────────┐
│  1. AUTHENTICATE  │  — Verify Supabase JWT Bearer token
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  2. SAVE + HASH   │  — Save to /tmp/uploads, compute SHA256
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  3. HASH CHECK    │  — Block if hash matches previously rejected content
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────────────────────────┐
│  4. PRIMARY AI — DEEPFAKE DETECTION (main.py)         │
│                                                       │
│   ① Metadata Scan    (weight: 10%)                   │
│   ② Frame Extraction (max 32 frames @ 1 FPS)         │
│   ③ Face Detection   (MTCNN)                          │
│   ④ EfficientNet-B0 ONNX + HuggingFace Ensemble      │
│   ⑤ Artifact Analysis (weight: 20%)                  │
│   ⑥ Temporal Analysis (weight: 15%)                  │
│   ⑦ Compression Analysis (weight: 15%)               │
│                                                       │
│   final_score = (0.40 × model) + (0.20 × artifact)  │
│               + (0.15 × temporal) + (0.10 × meta)   │
│               + (0.15 × compression)                 │
└────────────────────┬──────────────────────────────────┘
                     │
         ┌───────────┴────────────┐
         │                        │
    score < 0.60            score ≥ 0.60
    (APPROVED)           ┌──────────────────┐
         │               │  0.60 – 0.80     │  score ≥ 0.80
         │               │  UNDER_REVIEW    │  REJECTED ──────┐
         │               └────────┬─────────┘                 │
         ▼                        ▼                           │
┌─────────────────┐   ┌────────────────────┐                 │
│ 5. FAKE NEWS    │   │ Post stored but    │                 │
│ DETECTION       │   │ not visible on     │                 │
│ (context_verify │   │ public feed        │                 │
│  .py)           │   └────────────────────┘                 │
└────────┬────────┘                                          │
         │                                                   │
    ALLOW / BLOCK                                            │
         │                                                   ▼
         ▼                                        ┌──────────────────┐
┌─────────────────────────────────┐               │  6. BLOCK POST   │
│  6. COMPUTE FINAL VERDICT       │               │  Notify User     │
│                                 │               │  Track Alert     │
│  REAL       → Upload + Publish  │               │  Update Trust ↓  │
│  UNDER_REVIEW → Upload + Hold   │               └──────────────────┘
│  FAKE       → Block + Notify    │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  7. UPDATE TRUST SCORE          │
│                                 │
│  Verification Rate  (0-40 pts) │
│  Account Age        (0-20 pts) │
│  Activity Volume    (0-20 pts) │
│  Community Repute   (0-20 pts) │
└─────────────────────────────────┘
```

---

### 2️⃣ Community Reporting & Secondary Review Pipeline

When users report a post, TrueFrame uses a **weighted threshold system** to decide when to run a secondary AI model — one that uses completely different detection techniques than the primary.

```
User Reports a Post (trust_score ≥ 70 required)
        │
        ▼
┌──────────────────────────────────────────────────┐
│  WEIGHTED FLAG SCORING                           │
│                                                  │
│  flag_weight = reporter_trust_score / 100        │
│                                                  │
│  Example:                                        │
│  • User with trust 90 → weight 0.90             │
│  • User with trust 70 → weight 0.70             │
│                                                  │
│  weighted_sum = Σ(flag_weight for all flags)     │
│                                                  │
│  Threshold:                                      │
│  • Normal post  → 2.5 weighted sum required     │
│  • Restored post → 4.0 (prevents re-harassment) │
└─────────────────────┬────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │ weighted_sum < threshold │ weighted_sum ≥ threshold
         │                         │
         ▼                         ▼
   Notify owner only      ┌────────────────────────┐
   (if ≥ 3 raw flags)     │  TRIGGER SECONDARY     │
                          │  REVIEW                │
                          │                        │
                          │  • Post → UNDER_REVIEW │
                          │  • Yellow overlay shown│
                          │  • Feed hidden         │
                          └──────────┬─────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────┐
│  SECONDARY AI DETECTION (secondary_review.py)              │
│  (Completely different from primary pipeline)              │
│                                                            │
│  ① Frequency Spectrum Analysis      (weight: 30%)         │
│     — Azimuthal FFT, radial power spectral density        │
│     — GAN images deviate from natural 1/f distribution    │
│                                                            │
│  ② Cross-Patch Consistency          (weight: 25%)         │
│     — 16 overlapping face patches                         │
│     — GANs have uniform noise; real faces show variance   │
│                                                            │
│  ③ Noise Residual Analysis          (weight: 20%)         │
│     — SRM-style high-pass filters                         │
│     — Co-occurrence matrix analysis                       │
│                                                            │
│  ④ Edge Coherence Analysis          (weight: 15%)         │
│     — Multi-threshold Canny detection                     │
│     — Face-swap boundary discontinuity detection          │
│                                                            │
│  ⑤ EXIF Re-verification             (weight: 10%)         │
│     — Quantization table analysis                         │
│     — Editing software signature detection                │
│                                                            │
│  secondary_score = weighted fusion of all 5 components    │
└──────────────────────────────┬─────────────────────────────┘
                               │
              ┌────────────────┼─────────────────┐
              │                │                 │
         score < 0.60    0.60 – 0.80        score ≥ 0.80
              │                │                 │
              ▼                ▼                 ▼
     ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
     │   RESTORE    │  │   MANUAL     │  │    AUTO-REMOVE   │
     │              │  │   REVIEW     │  │                  │
     │ Post → PUBLIC│  │ Notify all   │  │ Delete post      │
     │ Dismiss flags│  │ verifiers    │  │ Delete from S3   │
     │ Notify owner │  │ Queue page   │  │ Confirm flags    │
     │ "Authentic"  │  │ shown        │  │ Notify owner     │
     └──────────────┘  └──────────────┘  │ Apply penalty   │
                                          │ Track alert     │
                                          └──────────────────┘

⚠️  FAIL-CLOSED: If AI script errors or times out → post stays UNDER_REVIEW

📊  TRUST PENALTY FOR REMOVAL:
    1st offense: -15 trust points
    2nd offense: -25 trust points
    3rd+ offense: -40 trust points (per removal)
```

---

### 3️⃣ Trust Score System

Every user has a dynamic **Trust Score (0–100)** that affects their content ranking, moderation privileges, and platform access.

```
┌─────────────────────────────────────────────────────────┐
│                  TRUST SCORE CALCULATION                │
│                                                         │
│  Component 1: Verification Rate          (0 – 40 pts)  │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━              │
│  verified_uploads / total_uploads × 40                  │
│  (highest weight — directly tied to authenticity)       │
│                                                         │
│  Component 2: Account Age               (0 – 20 pts)   │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━              │
│  min(20, account_age_days / 90 × 20)                    │
│  (maxes out at 90 days)                                 │
│                                                         │
│  Component 3: Activity Volume           (0 – 20 pts)   │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━              │
│  min(20, total_uploads / 50 × 20)                       │
│  (maxes out at 50 uploads)                              │
│                                                         │
│  Component 4: Community Reputation      (0 – 20 pts)   │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━              │
│  Followers (0-10): min(10, followers / 100 × 10)        │
│  Engagement (0-10): min(10, total_likes / 500 × 10)    │
│                                                         │
│  FINAL = min(100, sum of all components)                │
└─────────────────────────────────────────────────────────┘

TRUST STATUS TIERS:

  🆕 NEW_USER      → No uploads yet
  ✅ TRUSTED       → Score ≥ 50
  ⚠️  AT_RISK       → Score 25–49
  🔴 UNDER_REVIEW  → Score < 25

TRUST-GATED PRIVILEGES:

  Score ≥ 50  → Standard posting
  Score ≥ 70  → Can flag/report content
  Score ≥ 80  → Community verifier eligible
  Score ≥ 90  → Can make manual moderation decisions

FEED RANKING FORMULA:

  rankingScore = (0.60 × trustWeight)
               + (0.25 × recencyScore)
               + creatorBoost (0.15 if verified creator)
               + cameraBoost  (0.05 if camera capture)

  trustWeight: TRUSTED=1.0, AT_RISK=0.5, UNDER_REVIEW=0.1
  recencyScore: max(0, 1 − hours_old/72) — decays over 3 days
```

---

## 🗄️ Database Schema

### Core Tables

```
profiles                    verification_logs
━━━━━━━━━━━━━━━━━━━━        ━━━━━━━━━━━━━━━━━━━━━━━━━━
id (UUID, PK)               id (UUID, PK)
username (UNIQUE)           user_id → profiles
display_name                media_hash (SHA256)
avatar_url                  deepfake_verdict
bio                         fake_news_verdict
trust_status                final_verdict
trust_score (0-100)         score / final_score
real_percentage             model_name / model_version
fake_percentage             score_breakdown (JSONB)
total_attempts              authenticity_label
is_community_verifier       upload_source
identity_verified           created_at

posts                       secondary_reviews
━━━━━━━━━━━━━━━━━━━━        ━━━━━━━━━━━━━━━━━━━━━━━━━━
id (UUID, PK)               id (UUID, PK)
user_id → profiles          post_id → posts (UNIQUE)
media_url                   triggered_by
media_type                  trigger_flag_count
caption                     status (PENDING/PROCESSING
visibility (PUBLIC/         COMPLETED/FAILED)
  UNDER_REVIEW)             secondary_score
verification_status         frequency_score
authenticity_label          gan_artifact_score
like_count                  noise_consistency_score
comment_count               edge_coherence_score
share_count                 patch_variance_score
content_hash_proof          decision (RESTORE/
                              MANUAL_REVIEW/REMOVE)
community_flags             manual_decision
━━━━━━━━━━━━━━━━━━━━        manual_reviewer_id
id (UUID, PK)
post_id → posts             notifications
flagger_id → profiles       ━━━━━━━━━━━━━━━━━━━━━━━━━━
flag_type                   id (UUID, PK)
reason                      user_id → profiles
source_url                  type (LIKE/COMMENT/FOLLOW/
status                        VERIFICATION_PASSED/
flag_weight (0.0-1.0)         VERIFICATION_FAILED/
triggered_review              FLAG_RESULT/POST_REMOVED/
                              TRUST_PENALTY/...)
                            title / message
                            is_read
```

---

## 🤖 AI Detection Models

### Primary Detection (`main.py`)

| Component | Model | Weight |
|-----------|-------|--------|
| Neural Network | EfficientNet-B0 ONNX + HuggingFace Ensemble | 40% |
| Artifact Analysis | GAN fingerprint, compression heuristics | 20% |
| Temporal Analysis | Face consistency across video frames | 15% |
| Compression Analysis | JPEG quality degradation detection | 15% |
| Metadata Scan | EXIF integrity, file structure | 10% |

**Decision Thresholds:**
- `< 0.60` → ✅ APPROVED (Real content)
- `0.60 – 0.79` → 🟡 UNDER_REVIEW (Borderline)
- `≥ 0.80` → 🔴 REJECTED (Deepfake)

---

### Secondary Detection (`secondary_review.py`)

Triggered by community reports. Uses **fundamentally different techniques** to catch what the primary missed.

| Component | Technique | Weight |
|-----------|-----------|--------|
| Frequency Spectrum | Azimuthal FFT, radial power spectral density, GAN frequency drop-off | 30% |
| Cross-Patch Consistency | 16-patch variance analysis — GANs show uniform noise | 25% |
| Noise Residual | SRM-style high-pass filters, co-occurrence matrices | 20% |
| Edge Coherence | Multi-threshold Canny, face-swap boundary detection | 15% |
| EXIF Re-verification | Quantization tables, editing software signatures | 10% |

**Decision Thresholds:**
- `< 0.60` → ✅ RESTORE (Post returns to public feed)
- `0.60 – 0.79` → 🟡 MANUAL_REVIEW (Community verifiers decide)
- `≥ 0.80` → 🔴 REMOVE (Auto-deleted, trust penalized)

---

### Fake News Detection (`context_verify.py`)

Runs only if primary deepfake check passes.

| Component | Weight |
|-----------|--------|
| Temporal Consistency (media age vs "breaking now" claims) | 40% |
| Source Verification (trusted sources check) | 40% |
| Pattern Detection (sensationalist language) | 20% |

**Decision:** `ALLOW` / `BLOCK_UNVERIFIED` / `BLOCK_FAKE`

---

## 🔌 API Reference

### Upload & Verification
```
POST /api/upload/verify-upload    — Upload and verify media (multipart)
```

### Feed
```
GET  /api/feed/                   — Trust-weighted content feed (PUBLIC only)
```

### Community & Moderation
```
POST /api/community/flag/:postId           — Report a post (trust ≥ 70)
GET  /api/community/flags/:postId          — Get flags for a post
POST /api/community/flag/:flagId/review    — Review a flag (trust ≥ 90)
GET  /api/community/verifiers              — List community verifiers

GET  /api/moderation/pending               — Posts awaiting manual review
POST /api/moderation/review/:reviewId      — Submit RESTORE or REMOVE decision
```

### Social
```
POST /api/social/like/:postId              — Toggle like
GET  /api/social/like/:postId/status       — Check like status
POST /api/social/comment/:postId           — Add comment
GET  /api/social/comments/:postId          — Get comments
POST /api/social/share/:postId             — Track share
DELETE /api/social/post/:postId            — Delete own post
```

### Transparency
```
GET  /api/transparency/post/:postId        — Full verification details
GET  /api/transparency/timeline/:username  — User authenticity history
```

### Profile & Auth
```
GET  /api/profile/:username                — Public profile + posts
POST /api/profile/update                   — Update display name / bio
POST /api/profile/:username/follow         — Follow user
DELETE /api/profile/:username/follow       — Unfollow user

POST /api/verification/selfie              — Submit selfie for identity check
GET  /api/verification/status              — Check verification status

GET  /api/notifications/                   — Get notifications (paginated)
POST /api/notifications/read               — Mark notifications as read
GET  /api/notifications/unread-count       — Get unread count
```

---

## 📁 Project Structure

```
verified-stream/
│
├── src/                          ← React Frontend
│   ├── components/
│   │   ├── ui/                   ← shadcn-ui primitives
│   │   ├── feed/
│   │   │   └── PostCard.tsx      ← Post display + Under Review overlay
│   │   ├── upload/
│   │   │   └── UploadStep.tsx    ← Verification progress UI
│   │   ├── community/
│   │   │   └── FlagModal.tsx     ← Report content modal
│   │   ├── moderation/
│   │   │   └── ModerationQueue.tsx  ← Verifier review queue
│   │   ├── transparency/
│   │   │   └── TransparencyPanel.tsx  ← Verification details panel
│   │   ├── share/
│   │   │   └── ShareModal.tsx
│   │   └── layout/
│   │       ├── AppLayout.tsx
│   │       └── BottomNav.tsx
│   ├── pages/
│   │   ├── Feed.tsx
│   │   ├── Upload.tsx
│   │   ├── Dashboard.tsx
│   │   ├── Profile.tsx
│   │   ├── Moderation.tsx        ← Community verifier page
│   │   └── ...
│   ├── lib/
│   │   ├── supabase.ts           ← Supabase client (anon key)
│   │   └── api.ts                ← Backend URL constant
│   └── App.tsx                   ← Routes
│
├── backend/
│   ├── src/
│   │   ├── routes/
│   │   │   ├── upload.ts         ← Primary verification pipeline
│   │   │   ├── feed.ts           ← Trust-ranked feed
│   │   │   ├── community.ts      ← Reporting + flag threshold
│   │   │   ├── moderation.ts     ← Secondary review + decisions
│   │   │   ├── social.ts         ← Likes, comments, shares
│   │   │   ├── transparency.ts   ← Verification details API
│   │   │   ├── notifications.ts
│   │   │   ├── profile.ts
│   │   │   ├── dashboard.ts
│   │   │   ├── verification.ts   ← Identity (selfie) verification
│   │   │   └── ...
│   │   ├── lib/
│   │   │   ├── ai-runner.ts      ← Shared Python spawn utility
│   │   │   └── trust.ts          ← Shared trust score + alerts
│   │   ├── supabase.ts           ← Supabase client (service role)
│   │   └── index.ts              ← Server entry + route registration
│   └── db/
│       ├── schema.sql            ← Full database schema (V5)
│       └── migration_secondary_review.sql  ← Community review tables
│
└── ai_service/
    ├── main.py                   ← Primary deepfake detection
    ├── secondary_review.py       ← Secondary GAN/frequency analysis
    ├── context_verify.py         ← Fake news caption analysis
    ├── config.py                 ← Weights and thresholds
    ├── core/
    │   ├── models.py             ← EfficientNet ONNX + HuggingFace
    │   ├── detector.py           ← MTCNN face detection
    │   ├── heuristics.py         ← Artifact analysis
    │   ├── temporal.py           ← Video temporal consistency
    │   ├── compression.py        ← JPEG compression analysis
    │   ├── patches.py            ← Multi-patch voting
    │   ├── precheck.py           ← Metadata scanner
    │   └── extractor.py          ← Frame extraction
    └── models/
        └── efficientnet_b0_v1.onnx
```

---

## ⚙️ Environment Setup

### Frontend `.env.development`
```env
VITE_BACKEND_URL=http://localhost:3001
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
```

### Backend `backend/.env`
```env
PORT=3001
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_ANON_KEY=your-anon-key
DATABASE_URL=postgresql://...
PYTHON_PATH=python3
NODE_ENV=development
ALLOWED_ORIGIN=http://localhost:8080
```

### Development Commands

```bash
# Frontend (from verified-stream/)
npm run dev        # Vite dev server → http://localhost:8080
npm run build      # Production build
npm run lint       # ESLint

# Backend (from verified-stream/backend/)
npm run dev        # Fastify dev server → http://localhost:3001
npm run build      # Compile TS to dist/
npm run start      # Run compiled output

# AI Service (invoked automatically by backend)
python3 ai_service/main.py <file_path>
python3 ai_service/secondary_review.py <file_path>
python3 ai_service/context_verify.py <caption> <file_path>

# Database Migration (run in Supabase SQL editor)
# 1. Run backend/db/schema.sql (initial setup)
# 2. Run backend/db/migration_secondary_review.sql (community review)
```

---

## 🔄 Full User Journey

```
1. 🆕 User Signs Up
   └── Supabase Auth (Google/GitHub OAuth)
   └── Profile created with NEW_USER status

2. 📤 User Uploads Content
   └── File saved to /tmp → SHA256 hashed
   └── Primary AI pipeline runs (120s max)
   └── Result: APPROVED → Published ✅
              UNDER_REVIEW → Held 🟡
              REJECTED → Blocked 🔴

3. 📱 Feed Browsing
   └── Trust-ranked feed (only PUBLIC + APPROVED posts)
   └── Transparency panel available on every post
   └── Verified badge shows AI authentication details

4. 🚩 Community Reporting
   └── User flags suspicious post (trust ≥ 70 required)
   └── Weighted score computed
   └── Threshold reached → Secondary AI triggered
   └── Post goes UNDER_REVIEW (yellow overlay)

5. 🤖 Secondary Review
   └── Different AI techniques analyze the content
   └── RESTORE → Post returns to feed ✅
   └── MANUAL → Community verifiers decide 🟡
   └── REMOVE → Post deleted, trust penalized 🔴

6. 📊 Dashboard
   └── User sees verification history, trust score trend
   └── Community verifiers see moderation queue at /moderation
```

---

## 🛡️ Security & Trust Principles

- 🔒 **Service role key is backend-only** — never exposed to frontend
- 🔑 **JWT verified on every API call** — no session cookies
- 🔐 **Multi-tenant safety** — all queries scoped by user ID
- 🚫 **No PII in logs** — sensitive data never logged
- ⛔ **Fail-closed AI** — errors block content, never allow it
- 🗑️ **Temp files deleted** — uploaded files cleaned after processing
- 📋 **GDPR compliant** — identity verification data deletable
- 🧩 **Zod validation** — all API inputs validated at boundaries

---

*Built with ❤️ — because truth matters.*
