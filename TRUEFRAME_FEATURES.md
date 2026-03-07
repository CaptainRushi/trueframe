# TrueFrame — Complete Feature Documentation

> **The World's First AI-Powered Authentic Social Platform**
> Every photo and video verified real. No deepfakes, no manipulation, just truth.

---

## Platform Overview

TrueFrame is an AI-powered deepfake detection and content verification social platform. Users upload images and videos which pass through a multi-layer AI verification pipeline. Only content verified as "real" gets published to the social feed. The system is **fail-closed** — any error, timeout, or ambiguous result blocks the content.

**Website:** [https://vizora1.vercel.app](https://vizora1.vercel.app)
**Stack:** React + Vite (Frontend), Fastify + Node.js (Backend), Python + ONNX (AI Engine), Supabase (Database + Auth + Storage)

---

## Table of Contents

1. [Real Content Verification System](#1-real-content-verification-system)
2. [TrueFrame Trust Score](#2-trueframe-trust-score)
3. [Fake News Detection System](#3-fake-news-detection-system)
4. [Camera-Only Mode (Verified Capture)](#4-camera-only-mode-verified-capture)
5. [Authenticity Timeline](#5-authenticity-timeline)
6. [Content Transparency Panel](#6-content-transparency-panel)
7. [Community Fact-Checking](#7-community-fact-checking)
8. [AI Manipulation Detection](#8-ai-manipulation-detection)
9. [Reputation-Based Feed](#9-reputation-based-feed)
10. [Verified Creator Program](#10-verified-creator-program)
11. [AI Authenticity Dashboard](#11-ai-authenticity-dashboard)
12. [Blockchain Proof of Authenticity](#12-blockchain-proof-of-authenticity)
13. [Deepfake Alert System](#13-deepfake-alert-system)
14. [Notification System](#14-notification-system)
15. [Trust-Aware Comments](#15-trust-aware-comments)
16. [Social Features](#16-social-features)
17. [Account & Security](#17-account--security)
18. [Technical Architecture](#18-technical-architecture)

---

## 1. Real Content Verification System

Every post on TrueFrame goes through a **multi-layer Authenticity Verification Pipeline** before it can appear on the platform.

### How It Works

When a user uploads media, the following checks run sequentially:

| Step | Process | Description |
|------|---------|-------------|
| 1 | **Authentication** | Verify user identity via Supabase JWT |
| 2 | **File Processing** | Save to secure temp storage, compute SHA256 hash |
| 3 | **Duplicate Check** | Compare hash against previously rejected content |
| 4 | **Deepfake Detection** | Run multi-model AI analysis (see Feature 8) |
| 5 | **Fake News Check** | Analyze caption for misinformation patterns |
| 6 | **Verdict Computation** | Combine all scores into final verdict |
| 7 | **Action** | Approve and publish, or block and log |

### Authenticity Labels

Every piece of content receives one of these labels:

| Label | Badge | Meaning |
|-------|-------|---------|
| `CAMERA_ORIGINAL` | Captured on TrueFrame | Taken inside TrueFrame camera, no manipulation |
| `VERIFIED_REAL` | Verified Real | No manipulation detected via gallery upload |
| `EDITED` | Minor Edits | Minor edits like filters detected |
| `REJECTED_SYNTHETIC` | Suspicious | Deepfake or synthetic content detected |
| `REJECTED_MISLEADING` | Misleading | Fake news or misleading caption detected |

### Verdict Logic

| Final Score | Verdict | Action |
|-------------|---------|--------|
| < 0.40 | APPROVED | Published to feed |
| 0.40 - 0.59 | REJECTED (unsafe margin) | Blocked |
| >= 0.60 | REJECTED | Blocked |
| Any error/timeout | REJECTED | Blocked (fail-closed) |

### Fail-Closed Policy

TrueFrame operates on a **fail-closed** security model:
- If the AI engine crashes → upload blocked
- If verification times out (15s) → upload blocked
- If no face detected when expected → upload blocked
- If context engine errors → upload blocked

Safety over convenience. Always.

---

## 2. TrueFrame Trust Score

Every user has a **Trust Score from 0 to 100**, calculated from multiple weighted factors.

### Score Calculation

| Factor | Weight | Max Points | How It's Earned |
|--------|--------|------------|-----------------|
| Verification Rate | 40% | 40 pts | Ratio of approved vs total uploads |
| Account Age | 20% | 20 pts | Maxes out at 90 days |
| Activity Volume | 20% | 20 pts | Maxes out at 50 uploads |
| Community Reputation | 20% | 20 pts | Followers (10 pts) + Post engagement (10 pts) |

### Trust Levels

| Score Range | Level | Visual |
|-------------|-------|--------|
| 90 - 100 | Trusted Creator | Green shield |
| 70 - 89 | Reliable User | Green shield |
| 50 - 69 | Normal User | Blue shield |
| 30 - 49 | Suspicious | Yellow shield |
| 0 - 29 | High Risk | Red shield |

### Trust Status (Account-Level)

| Status | Trigger | Effect |
|--------|---------|--------|
| `NEW_USER` | 0 uploads | Default state |
| `TRUSTED` | Score >= 50, fake rate < 10% | Full platform access |
| `AT_RISK` | Score 25-49, fake rate 10-30% | Warning displayed |
| `RESTRICTED` | Score < 25, fake rate > 30% | Limited visibility, posts excluded from Explore |

### Profile Display

Every profile shows:
- Trust Score (circular gauge out of 100)
- Trust Level label
- Verification Rate breakdown bar
- Account Age score bar
- Activity Volume score bar
- Community Reputation score bar
- Real vs Fake percentage bars
- Total / Verified / Fake upload counts

---

## 3. Fake News Detection System

TrueFrame checks text captions for misinformation before publishing.

### AI Analysis

The context verification engine (`context_verify.py`) analyzes:
- Misinformation patterns and sensational claims
- Unreliable source indicators
- Fact-checking keyword matching
- Claim credibility scoring

### Result Labels

| Verdict | Action |
|---------|--------|
| `ALLOW` | Caption is credible, publish |
| `BLOCK_FAKE` | Misleading factual claim detected, block |
| `BLOCK_UNVERIFIED` | Unverifiable claims, block |

### Comment-Level Detection

Comments are also classified:
- `NORMAL` — Regular comment
- `CORRECTION` — Contains debunking language
- `QUESTION` — Asks a question
- `CLAIM` — Contains factual claims (collapsed if from low-trust users)

Spam-like comments with promotional keywords are auto-blocked.

---

## 4. Camera-Only Mode (Verified Capture)

Users can upload media in **two modes**, with different verification levels.

### Mode 1 — Verified Capture (Camera)

| Feature | Detail |
|---------|--------|
| How | Photo taken inside TrueFrame's built-in camera |
| Badge | "Captured on TrueFrame" |
| Metadata | Signed with device info (user agent, timestamp, source) |
| Trust | Easier to achieve `CAMERA_ORIGINAL` label |
| Feed Boost | +5% ranking boost in feed algorithm |

**User Flow:**
1. Tap "Verified Capture" toggle on Upload page
2. Live camera viewfinder appears with "LIVE" indicator
3. Tap capture button to take photo
4. Add caption and submit for verification
5. If approved, post receives "Captured on TrueFrame" badge

### Mode 2 — Gallery Upload

| Feature | Detail |
|---------|--------|
| How | Upload from device gallery |
| Badge | "Verified Real" (if approved) |
| Metadata | Standard file metadata only |
| Trust | Requires stricter score thresholds |
| Feed Boost | Standard ranking |

**Requires extra verification** since the file could have been manipulated before upload.

---

## 5. Authenticity Timeline

Every user profile displays a **transparency timeline** showing their verification history.

### What It Shows

| Metric | Description |
|--------|-------------|
| Authentic | Posts that passed all verification |
| Edited | Posts with minor edits detected |
| Blocked | Posts that were rejected |
| Authenticity Rate | Percentage bar (authentic / total) |
| Daily Chart | Mini bar chart showing daily authentic vs blocked uploads |

### Time Periods

Users can toggle between:
- **7 days** — Recent activity
- **30 days** — Monthly overview (default)
- **90 days** — Quarterly view

### Example Display

```
Last 30 Days
Authentic: 28     Edited: 2     Blocked: 0
Authenticity Rate: 93%
[||||||||||||||||||||  ] 93%
```

---

## 6. Content Transparency Panel

When someone taps the **eye icon** on any post, they see full verification analysis.

### Panel Contents

| Section | Data Shown |
|---------|------------|
| **Authenticity Score** | Overall score as percentage with color-coded bar |
| **Deepfake Probability** | Percentage from neural network model |
| **AI Generated Probability** | Percentage from artifact analysis |
| **Metadata Integrity** | Valid / Unknown |
| **Upload Type** | Camera Capture or Gallery Upload |
| **Detection Breakdown** | 4 individual analyzer scores with progress bars |
| **Verification Details** | Model name, version, verification timestamp |
| **Content Proof** | Blockchain-like hash proof with chain index |
| **Community Flags** | Number of flags and confirmed flags |
| **Author Info** | Username, trust score, creator status |

### Detection Breakdown Components

| Analyzer | What It Checks |
|----------|---------------|
| Neural Network | EfficientNet ONNX model deepfake probability (weight: 0.45) |
| Artifact Analysis | Visual artifacts, GAN fingerprints (weight: 0.25) |
| Temporal Consistency | Frame-to-frame consistency for video (weight: 0.20) |
| Metadata Scan | EXIF data, compression artifacts (weight: 0.10) |

### Example Panel

```
Authenticity Score: 96%
Deepfake Probability: 1%
AI Generated Probability: 3%
Metadata Integrity: Valid
Upload Type: Camera Capture

Detection Breakdown:
  Neural Network     ████░░░░░░  4%
  Artifact Analysis  ███░░░░░░░  3%
  Temporal Check     ██░░░░░░░░  2%
  Metadata Scan      █░░░░░░░░░  1%

Model: efficientnet-b0 v1.0
Verified: Mar 6, 2026 at 2:15 PM

Proof Hash: a3f8c2d1e5b7...
Chain Index: #1247
```

---

## 7. Community Fact-Checking

Users with high trust scores can become **Community Verifiers** who help maintain content integrity.

### Who Can Flag Content

| Requirement | Threshold |
|-------------|-----------|
| Trust Score | 70+ |
| OR | Community Verifier status |

### Flag Types

| Type | Description |
|------|-------------|
| `MISINFORMATION` | Contains false or misleading claims |
| `MANIPULATED` | Image/video has been altered |
| `OUT_OF_CONTEXT` | Real content used misleadingly |
| `SPAM` | Promotional or repetitive content |
| `OTHER` | Other concern |

### Flagging Flow

1. Tap the **flag icon** on any post
2. Select flag type
3. Add optional details and fact-check source URL
4. Submit flag

### Auto-Actions

| Condition | Action |
|-----------|--------|
| 3+ flags on a post | Post owner receives notification |
| Flag confirmed by senior verifier | Flag status updated to CONFIRMED |
| False flag pattern | Flagger trust score impacted |

### Flag Review

Senior verifiers (trust score 90+ or Community Verifier status) can review and confirm/dismiss flags.

---

## 8. AI Manipulation Detection

TrueFrame's AI engine detects multiple types of manipulation:

### Detection Capabilities

| Manipulation Type | Detection Method |
|-------------------|-----------------|
| Face Swap | EfficientNet neural network + face region analysis |
| GAN Generated Faces | Artifact pattern detection |
| Photoshop Edits | Compression inconsistency analysis |
| AI Generated Images | Model fingerprint detection |
| Voice Cloning | Audio pattern analysis (video) |
| Lip Sync Deepfake | Temporal frame consistency checking |
| Metadata Tampering | EXIF data validation |
| Re-compression | JPEG quality level analysis |

### Scoring Model

The final detection score is a **weighted fusion** of multiple analyzers:

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| EfficientNet ONNX Model | 0.45 | Primary deepfake probability |
| Artifact Analysis | 0.25 | Visual anomaly patterns |
| Temporal Consistency | 0.20 | Frame coherence (video) |
| Metadata Scan | 0.10 | File integrity checks |

### Technology

- **Model:** EfficientNet-B0 (ONNX runtime)
- **Face Detection:** MTCNN (Multi-task Cascaded Convolutional Networks)
- **Image Processing:** OpenCV
- **Runtime:** Python with ONNX Runtime

If any manipulation is detected above threshold → **upload blocked immediately**.

---

## 9. Reputation-Based Feed

The TrueFrame feed is **not chronological** — it's ranked by trust and authenticity.

### Ranking Algorithm

| Factor | Weight | Description |
|--------|--------|-------------|
| Author Trust | 60% | Based on author's fake content rate |
| Recency | 25% | Decays to 0 over 72 hours |
| Verified Creator Boost | +15% | If author is a Verified Creator |
| Camera Capture Boost | +5% | If post was captured in TrueFrame camera |

### Trust Weight Mapping

| Author Fake Rate | Trust Weight |
|------------------|-------------|
| < 10% | 1.0 (full trust) |
| 10% - 30% | 0.5 (reduced) |
| > 30% | 0.1 (heavily penalized) |

### What This Means

- High-trust creators' posts appear at the top
- Verified content is prioritized
- Camera-captured posts get a slight boost
- Fake content uploaders are naturally suppressed
- Viral but untrustworthy content doesn't spread

### Explore Feed Rules

The Explore page has additional filters:
- Only `PUBLIC` visibility posts
- `RESTRICTED` users' posts are excluded
- Sorted by recency with trust filtering

---

## 10. Verified Creator Program

Creators who maintain an exceptional authenticity record earn the **Verified Authentic Creator** badge.

### Eligibility Requirements

| Criteria | Threshold |
|----------|-----------|
| Trust Score | 85+ |
| Total Uploads | 10+ verified uploads |
| Authenticity Rate | 90%+ |
| Identity Verified | Recommended (not required) |

### How to Apply

1. Navigate to **Creator Program** page (from Dashboard)
2. View real-time eligibility tracker with progress bars
3. When all criteria are met, tap "Claim Creator Badge"
4. Badge is granted instantly

### Creator Benefits

| Benefit | Description |
|---------|-------------|
| Verified Authentic Creator Badge | Displayed on profile and all posts |
| Higher Feed Visibility | +15% ranking boost in feed algorithm |
| Priority Verification | Faster processing for uploads |
| Community Verifier Status | Auto-granted, can flag content |
| Authenticity Badge on Posts | Visual indicator on every post |

### Creator Leaderboard

Public leaderboard showing top creators ranked by:
- Trust Score (primary)
- Authenticity Rate
- Total verified uploads
- Verified Creator status

Each entry shows rank (with crown icons for top 3), avatar, username, trust level label, trust score, and authenticity percentage.

---

## 11. AI Authenticity Dashboard

Every user has a personal dashboard showing comprehensive verification analytics.

### Dashboard Sections

#### Trust Score Card
- Large numeric display (out of 100)
- Animated circular gauge
- Trust status label and description

#### Trust Level Indicator
All 5 trust levels displayed with the user's current level highlighted:
- Trusted Creator (90-100)
- Reliable User (70-89)
- Normal User (50-69)
- Suspicious (30-49)
- High Risk (0-29)

#### Statistics Grid
- **Verified Real** — Count of approved uploads
- **Fake Uploads** — Count of rejected uploads with trend indicator

#### Trust Score Over Time
- Line/area chart showing trust score history
- Up to 90 days of historical data
- Green gradient visualization

#### Content Authenticity
- Real Content percentage bar (green)
- Fake Content percentage bar (red)
- Summary: Total Attempts / Real / Fake counts

#### Recent Verification Logs
- Scrollable history of upload attempts
- Each entry shows: verdict (Approved/Blocked), date, reason/score
- Color-coded icons (green for approved, red for blocked)

#### Creator Program CTA
- Real-time eligibility checker
- Progress bars for each requirement
- Direct link to Creator Program page

---

## 12. Blockchain Proof of Authenticity

Every verified post receives a **cryptographic proof** that creates an immutable chain.

### How It Works

| Step | Process |
|------|---------|
| 1 | When a post passes verification, a **content proof** is generated |
| 2 | The media file's SHA256 hash is combined with metadata |
| 3 | The previous proof's hash is chained to create a new proof hash |
| 4 | The proof is stored with a chain index number |

### Proof Structure

| Field | Description |
|-------|-------------|
| `media_hash` | SHA256 hash of the original media file |
| `metadata_hash` | SHA256 of post ID + user ID + timestamp |
| `proof_hash` | SHA256 of (previous_proof + media_hash + metadata_hash) |
| `previous_proof_hash` | Links to the previous proof in the chain |
| `proof_chain_index` | Sequential index in the global proof chain |

### Verification

Anyone can verify a post's authenticity by:
1. Opening the Content Transparency Panel
2. Viewing the Proof of Authenticity section
3. Checking the proof hash and chain index

### What This Proves

- The content has **not been modified** since verification
- The verification occurred at a **specific point in time**
- The proof is **linked** to all previous proofs (chain integrity)
- The original file hash can be **independently verified**

---

## 13. Deepfake Alert System

TrueFrame tracks and alerts users about widespread deepfake content.

### How Alerts Are Generated

| Trigger | Action |
|---------|--------|
| First detection of a media hash | Alert created with severity `LOW` |
| Same hash detected 3+ times | Severity upgraded to `MEDIUM` |
| Same hash detected 5+ times | Severity upgraded to `HIGH` |
| Same hash detected 10+ times | Severity upgraded to `CRITICAL` |

### Alert Severity Levels

| Level | Color | Meaning |
|-------|-------|---------|
| LOW | Yellow | Isolated detection |
| MEDIUM | Yellow | Multiple detection attempts |
| HIGH | Orange | Widespread circulation attempt |
| CRITICAL | Red | Viral deepfake content detected |

### Where Alerts Appear

- **Notifications page** → "Deepfake Alerts" tab
- Each alert shows: title, description, severity badge, detection count, timestamp

### Example Alert

```
CRITICAL
Deepfake Content Detected
This viral video is confirmed as deepfake
Detected 12x | Mar 6, 2026
```

---

## 14. Notification System

TrueFrame has a comprehensive real-time notification system.

### Notification Types

| Type | Trigger | Icon |
|------|---------|------|
| `LIKE` | Someone likes your post | Heart |
| `COMMENT` | Someone comments on your post | Message |
| `FOLLOW` | Someone follows you | User Plus |
| `VERIFICATION_PASSED` | Your upload passed verification | Green Shield |
| `VERIFICATION_FAILED` | Your upload was blocked | Red Shield |
| `TRUST_CHANGE` | Your trust level changed | Shield |
| `CREATOR_BADGE` | You earned Verified Creator badge | Award |
| `FLAG_RESULT` | Your post received community flags | Flag |
| `DEEPFAKE_ALERT` | Deepfake content alert | Alert |
| `SYSTEM` | Platform announcements | Bell |

### Features

- **Unread count badge** on bottom navigation bar (polls every 30 seconds)
- **Mark all as read** button
- **Today / Earlier** grouping
- **Rich notifications** with user avatars and post links
- **Unread indicator** dot on individual notifications

---

## 15. Trust-Aware Comments

Comments on TrueFrame are classified and filtered based on the commenter's trust level.

### Comment Classification

| Type | Trigger Keywords | Example |
|------|-----------------|---------|
| `NORMAL` | Default | "Great photo!" |
| `CORRECTION` | false, fake, wrong, debunk, misleading | "This is actually false because..." |
| `QUESTION` | why, how, what, ?, can you | "Where was this taken?" |
| `CLAIM` | percent, million, confirmed, proven | "Studies show 90% of..." |

### Visibility Rules

| Comment Type | Author Trust | Visibility |
|--------------|-------------|------------|
| CLAIM | TRUSTED | `VISIBLE` |
| CLAIM | Not TRUSTED | `COLLAPSED` |
| CLAIM + Spam keywords | Any | `BLOCKED` |
| NORMAL/CORRECTION/QUESTION | Any | `VISIBLE` |

### Anti-Spam

Comments containing promotional keywords (`click`, `link`, `buy`, `crypto`, `invest`, `prize`) in claim-type comments are automatically blocked.

---

## 16. Social Features

### Feed
- Trust-weighted algorithm (not pure chronological)
- Verified badge on every post
- Authenticity label overlay on post images
- Like, comment, share, bookmark actions
- Post deletion (owner only)

### Profiles
- Avatar upload (max 300KB, JPG/PNG/WEBP)
- Display name and bio editing
- Follower/Following counts with auto-updating triggers
- Posts grid and Reels grid (media type tabs)
- Trust score gauge with breakdown
- Authenticity Timeline
- Verified Creator badge
- Follow/Unfollow functionality

### Explore
- Grid layout of verified public content
- User and post search with debounced input
- Trust status badges on search results
- Restricted users excluded from results

### Sharing
- Share modal for verified content
- Share tracking (logged to `shares` table)
- Unverified content cannot be shared

---

## 17. Account & Security

### Authentication
- **Supabase Auth** with Google and GitHub OAuth
- JWT token verification on every API call
- Service role key is backend-only
- Anon key is frontend-only

### Sign Out
- Server-side session invalidation
- Frontend session cleanup
- Graceful handling of expired tokens

### Account Deletion
- **Full data purge** — irreversible
- Requires typing "DELETE MY ACCOUNT" to confirm
- Deletion order:
  1. Social interactions (likes, comments, shares)
  2. Follow relationships
  3. Storage files (avatars, post media)
  4. Posts
  5. Verification logs
  6. Profile record
  7. Auth user (prevents future login)

### Security Measures
- All uploads scanned before publishing
- Hash deduplication prevents re-uploading known fakes
- Trust score penalizes repeat offenders
- Restricted users have limited visibility
- Fail-closed verification prevents bypassing

---

## 18. Technical Architecture

### System Overview

```
User → Frontend (React/Vite) → Backend (Fastify)
                                    ↓
                              AI Service (Python)
                              - EfficientNet ONNX
                              - MTCNN Face Detection
                              - Artifact Analysis
                              - Temporal Consistency
                              - Metadata Scan
                                    ↓
                              Supabase
                              - PostgreSQL (data)
                              - Auth (identity)
                              - Storage (media)
```

### Frontend Stack

| Technology | Purpose |
|-----------|---------|
| React 18 | UI framework |
| Vite + SWC | Build tool |
| TypeScript | Type safety |
| TanStack React Query | Server state management |
| React Router v6 | Client-side routing |
| Tailwind CSS | Styling (dark-mode-first) |
| Framer Motion | Animations |
| Recharts | Dashboard charts |
| Lucide Icons | Icon system |
| shadcn/ui | UI component library |

### Backend Stack

| Technology | Purpose |
|-----------|---------|
| Fastify | HTTP server |
| TypeScript | Type safety |
| Supabase JS | Database & auth client |
| @fastify/multipart | File upload handling (50MB limit) |
| @fastify/cors | Cross-origin requests |
| child_process (spawn) | AI service execution |
| crypto | SHA256 hashing |

### AI Service Stack

| Technology | Purpose |
|-----------|---------|
| Python 3 | Runtime |
| ONNX Runtime | Model inference |
| EfficientNet-B0 | Primary deepfake detection model |
| MTCNN | Face detection |
| OpenCV | Image processing |
| Pillow | Image manipulation |

### Database Schema

| Table | Purpose |
|-------|---------|
| `profiles` | User profiles with trust data |
| `verification_logs` | Every upload attempt and result |
| `posts` | Only verified-real published content |
| `likes` | Post likes |
| `comments` | Trust-classified comments |
| `shares` | Share tracking |
| `follows` | Follow relationships |
| `trust_score_history` | Trust score snapshots over time |
| `community_flags` | Community fact-checking flags |
| `notifications` | User notification inbox |
| `content_proofs` | Blockchain-like hash chain |
| `deepfake_alerts` | Global deepfake alert tracking |

### API Routes

| Prefix | Module | Endpoints |
|--------|--------|-----------|
| `/api/upload` | Upload & Verification | `POST /verify-upload` |
| `/api/feed` | Feed | `GET /` |
| `/api/dashboard` | Dashboard | `GET /stats`, `GET /trend`, `GET /history` |
| `/api/profile` | Profiles | `GET /:username`, `POST /init`, `POST /update`, `POST /avatar`, `GET /me/history`, `POST /:username/follow`, `DELETE /:username/follow` |
| `/api/social` | Social | `POST /like/:id`, `GET /like/:id/status`, `POST /comment/:id`, `GET /comments/:id`, `POST /share/:id`, `DELETE /post/:id` |
| `/api` | Explore & Search | `GET /explore`, `GET /search/users`, `GET /search/posts` |
| `/api/account` | Account | `POST /signout`, `DELETE /` |
| `/api/transparency` | Transparency | `GET /post/:id`, `GET /timeline/:username` |
| `/api/community` | Community | `POST /flag/:id`, `GET /flags/:id`, `POST /flag/:id/review`, `GET /verifiers` |
| `/api/notifications` | Notifications | `GET /`, `POST /read`, `GET /unread-count`, `GET /alerts` |
| `/api/creator` | Creator Program | `GET /status`, `POST /apply`, `GET /leaderboard` |

### Environment Variables

**Frontend:**
| Variable | Purpose |
|----------|---------|
| `VITE_BACKEND_URL` | Backend API URL |
| `VITE_SUPABASE_URL` | Supabase project URL |
| `VITE_SUPABASE_ANON_KEY` | Supabase anon/public key |

**Backend:**
| Variable | Purpose |
|----------|---------|
| `PORT` | Server port (default 3001) |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key |
| `SUPABASE_ANON_KEY` | Supabase anon key |
| `DATABASE_URL` | Direct PostgreSQL connection |
| `REDIS_URL` | Redis cache URL |
| `PYTHON_PATH` | Path to Python binary |
| `NODE_ENV` | Environment (development/production) |
| `ALLOWED_ORIGIN` | CORS allowed origins |

---

## Page Map

| Route | Page | Auth Required |
|-------|------|---------------|
| `/` | Landing Page | No |
| `/login` | Login (OAuth) | No |
| `/feed` | Truth Feed | Yes |
| `/explore` | Explore & Search | Yes |
| `/upload` | Upload & Verify | Yes |
| `/dashboard` | Trust Dashboard | Yes |
| `/profile/:username?` | User Profile | Yes |
| `/notifications` | Activity & Alerts | Yes |
| `/creator` | Creator Program | Yes |
| `/post/:postId` | Single Post View | No |

---

## Design System

### Color Palette

| Token | Value | Usage |
|-------|-------|-------|
| Background Primary | `#0a0a0f` | Main background |
| Background Card | `#111118` | Card surfaces |
| Accent Blue | `#3b82f6` | Primary actions |
| Accent Green | `#10b981` | Verified/success states |
| Accent Red | `#ef4444` | Destructive/fake states |
| Accent Yellow | `#f59e0b` | Warning/at-risk states |
| Accent Cyan | `#06b6d4` | Info states |
| Text Primary | `#f1f5f9` | Main text |
| Text Muted | `#64748b` | Secondary text |
| Border | `#1e293b` | Card borders |

### Design Principles

- **Dark-mode first** — All UI designed for dark backgrounds
- **Rounded corners** — 2xl to 3xl border radius for cards
- **Glass morphism** — Backdrop blur for overlays and navigation
- **Micro-animations** — Framer Motion for meaningful transitions
- **Trust-first visual hierarchy** — Trust indicators are prominent
- **WCAG 2.1 AA** — Minimum accessibility standard

---

*Built with integrity. Every pixel verified.*
*TrueFrame — Where authenticity is the algorithm.*
