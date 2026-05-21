from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
import sys
import uuid
import json

# Ensure we can import from core
sys.path.append(os.path.dirname(__file__))

from main import analyze, _is_video
from training.reel_inference import analyze_video

app = FastAPI(title="Trueframe AI Detection Service")

# Ensure temp directory exists
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/health")
async def health_check():
    return {"status": "online", "model": "trueframe-signal-analyzer"}

@app.post("/verify")
async def verify_media(file: UploadFile = File(...)):
    """
    Accepts an image or video file, runs the detection pipeline, and returns the verdict.
    """
    # Create a unique filename to avoid collisions
    file_ext = os.path.splitext(file.filename)[1]
    temp_filename = f"{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join(TEMP_DIR, temp_filename)
    
    try:
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run detection
        if _is_video(temp_path):
            report = analyze_video(temp_path)
        else:
            report = analyze(temp_path)
        
        return report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    # Use port 8000 (standard for Lightning AI exposure)
    uvicorn.run(app, host="0.0.0.0", port=8000)
