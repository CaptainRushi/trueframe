# How to Run Locally

Since you have removed the Render backend, follow these steps to run everything on your local machine.

### 1. Prerequisites
- **Node.js** (v18+)
- **Supabase Account** (Ensure your `.env` files are updated with your Supabase credentials)

### 2. Setup Backend
1. Open a terminal in `backend/`
2. Create/Update `backend/.env`:
   ```env
   PORT=3001
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
   ```
3. Run the backend:
   ```bash
   npm install
   npm run dev
   ```
   *The backend should say: `Backend is live at: http://[::]:3001`*

### 3. Setup Frontend
1. Open a terminal in the root directory
2. Create/Update `.env.development`:
   ```env
   VITE_BACKEND_URL=http://localhost:3001
   VITE_SUPABASE_URL=your_supabase_url
   VITE_SUPABASE_ANON_KEY=your_anon_key
   ```
3. Run the frontend:
   ```bash
   npm install
   npm run dev
   ```
   *The frontend should be at: `http://localhost:8080`*

### 4. Common Fixes for "ERR_CONNECTION_REFUSED"
- **Check Port:** Ensure no other app is using port 3001.
- **Protocol:** Make sure you are using `http://` and NOT `https://` for the backend URL.
- **Wait for Startup:** Ensure the backend terminal shows the `[SUCCESS]` message before refreshing the frontend.
- **Browser Cache:** Hard refresh the browser (`Ctrl + F5` or `Cmd + Shift + R`).
