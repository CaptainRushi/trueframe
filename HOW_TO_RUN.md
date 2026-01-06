# How to Run (REAL-TIME VERIFICATION)

## Prerequisites
1. **Node.js** v18+
2. **Python** v3.8+
3. **NO Redis needed** (removed async workers)

---

## Start the System

### Terminal 1: Frontend (React)
```powershell
cd verified-stream
npm install
npm run dev
```
Runs at: **http://localhost:8080**

### Terminal 2: Backend API (Fastify)
```powershell
cd verified-stream/backend
npm install
npm run dev
```
Runs at: **http://localhost:3001**

---

## How It Works

1. User uploads image/video at `/upload`
2. Frontend sends file to `POST /api/upload/verify`
3. Backend:
   - Saves to `/tmp/uploads`
   - Spawns Python AI process **inline**
   - Waits for verification result (< 3 seconds)
   - Returns `verified: true/false`
4. Frontend:
   - If approved → allows caption + publish
   - If rejected → shows reason, deletes file

**No queues. No workers. No fake data.**

---

## Test Upload
1. Open http://localhost:8080/upload
2. Select any image or video
3. Click "Verify & Upload"
4. Watch real-time AI verification
5. See instant approval/rejection

---

## Fail-Closed Rules
- Python error → Block
- Timeout (>10s) → Block
- Score ≥ 0.40 → Block
- No face detected → Block
