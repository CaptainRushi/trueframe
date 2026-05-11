# ⚡ Hosting Trueframe AI on Lightning AI

This guide explains how to set up and host the Trueframe AI detection engine as a persistent API on Lightning AI Studios.

## 1. Setup Lightning Studio
1.  Go to [Lightning.ai](https://lightning.ai/) and log in.
2.  Click **"Create Studio"**.
3.  Choose the **"Python"** or **"PyTorch"** template.
4.  **Hardware Selection**: 
    - For development/testing: **CPU** (2 vCPU) is enough.
    - For production/performance: **GPU** (T4 or A10G) is highly recommended for the HuggingFace models.

## 2. Clone and Install
Open the terminal in your Lightning Studio and run:

```bash
# 1. Clone your repository
git clone <your-repo-url>
cd Trueframe-1/verified-stream/ai_service

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install API dependencies (FastAPI + Uvicorn)
pip install fastapi uvicorn python-multipart
```

## 3. Launch the API Server
Run the `server.py` file I created for you:

```bash
python server.py
```

You should see:
`INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)`

## 4. Expose the Port
To make the API accessible to your backend:
1.  In the Lightning Studio sidebar, click on the **"Network"** (globe icon) tab.
2.  Click **"Add New Port"**.
3.  Enter Port Number: `8000`.
4.  Lightning will generate a public URL for you. It will look like:
    `https://8000-01-xxxx-xxxx.lightning.ai`

## 5. Update Your Backend
In your `backend/.env` file, update the following:

```env
# Point this to your new Lightning AI URL
AI_SERVICE_URL=https://8000-01-xxxx-xxxx.lightning.ai/verify
```

## 6. Optimization Tips
- **Pre-download Models**: Run `python server.py` once to allow it to download the HuggingFace models (`dima806/deepfake_vs_real_image_detection`) into the Studio's persistent storage.
- **Auto-Restart**: Use `pm2` or a simple bash loop to keep the server running if it crashes.
- **Scaling**: Lightning allows you to "Duplicate" Studios to scale horizontally if you get a lot of traffic.
