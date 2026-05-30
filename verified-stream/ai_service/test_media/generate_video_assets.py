import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
SRC_VIDEO_PATH = PROJECT_ROOT / "176527-855920754_medium.mp4"
OUTPUT_DIR = SCRIPT_DIR / "videos"

print(f"Source video: {SRC_VIDEO_PATH}")
print(f"Output directory: {OUTPUT_DIR}")

# Haar Cascade for face detection
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

if face_cascade.empty():
    print("Error: Could not load Haar cascade face detector!")
    sys.exit(1)

def get_face_box(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for nb in [3, 2]:
        faces = face_cascade.detectMultiScale(gray, 1.05, nb, minSize=(32, 32))
        if len(faces) > 0:
            return sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    return None

def apply_face_swap(frame, face_img, box):
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return frame
    
    src_face = cv2.resize(face_img, (w, h))
    
    # Elliptical mask for smooth blending
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.45), int(h * 0.55)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 5)
    mask = np.expand_dims(mask, axis=-1)
    
    dst_frame = frame.copy()
    face_region = dst_frame[y:y+h, x:x+w]
    
    if face_region.shape[:2] == (h, w):
        blended = src_face * mask + face_region * (1.0 - mask)
        dst_frame[y:y+h, x:x+w] = blended.astype(np.uint8)
        
    return dst_frame

def apply_lip_sync(frame, box):
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return frame
    
    dst_frame = frame.copy()
    # Mouth area: bottom third of the face, centered
    my_start = y + int(h * 0.65)
    my_end = y + int(h * 0.95)
    mx_start = x + int(w * 0.22)
    mx_end = x + int(w * 0.78)
    
    mw = mx_end - mx_start
    mh = my_end - my_start
    
    if mw > 0 and mh > 0 and my_end <= frame.shape[0] and mx_end <= frame.shape[1]:
        mouth = dst_frame[my_start:my_end, mx_start:mx_end]
        # Pixelate by scaling down and up
        small = cv2.resize(mouth, (max(1, mw // 8), max(1, mh // 8)), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (mw, mh), interpolation=cv2.INTER_NEAREST)
        pixelated_blurred = cv2.GaussianBlur(pixelated, (5, 5), 1.5)
        dst_frame[my_start:my_end, mx_start:mx_end] = pixelated_blurred
        
    return dst_frame

def generate_video(input_path, output_path, manipulation_type, face_img=None, duration=5.0, compress=False):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Cannot open source video {input_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_width = width
    out_height = height
    if compress:
        # Resize to 1/3 resolution for heavy compression simulation
        out_width = max(160, width // 3)
        out_height = max(120, height // 3)
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height))
    
    max_frames = int(fps * duration)
    frame_count = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        box = get_face_box(frame)
        
        if box is not None:
            if manipulation_type == "real":
                modified_frame = frame
            elif manipulation_type == "face_swap" and face_img is not None:
                modified_frame = apply_face_swap(frame, face_img, box)
            elif manipulation_type == "lip_sync":
                modified_frame = apply_lip_sync(frame, box)
            else:
                modified_frame = frame
        else:
            modified_frame = frame
            
        if compress:
            modified_frame = cv2.resize(modified_frame, (out_width, out_height))
            
        out.write(modified_frame)
        frame_count += 1
        
    cap.release()
    out.release()
    print(f"Generated {output_path} ({frame_count} frames)")
    return True

def main():
    if not SRC_VIDEO_PATH.exists():
        print(f"Error: Source stock video not found at {SRC_VIDEO_PATH}")
        sys.exit(1)
        
    # Load StyleGAN images for face swapping
    face_imgs = []
    stylegan_dir = PROJECT_ROOT / "test_assets" / "stylegan_tpdne"
    for i in range(1, 4):
        p = stylegan_dir / f"tpdne_0{i}.jpg"
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                face_imgs.append(img)
                
    if not face_imgs:
        print("Warning: No StyleGAN images found. Creating synthesized dummy face.")
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.circle(dummy, (128, 128), 90, (180, 200, 255), -1)
        cv2.circle(dummy, (90, 100), 10, (50, 50, 50), -1)
        cv2.circle(dummy, (166, 100), 10, (50, 50, 50), -1)
        cv2.ellipse(dummy, (128, 170), (40, 20), 0, 0, 180, (50, 50, 255), 3)
        face_imgs.append(dummy)
        
    real_dir = OUTPUT_DIR / "real"
    fake_dir = OUTPUT_DIR / "deepfake"
    
    # 1. Real videos (V01-V05)
    print("\n--- Generating Real Videos ---")
    generate_video(SRC_VIDEO_PATH, real_dir / "v01_real_speech.mp4", "real")
    generate_video(SRC_VIDEO_PATH, real_dir / "v02_real_speech.mp4", "real")
    generate_video(SRC_VIDEO_PATH, real_dir / "v03_real_speech.mp4", "real")
    generate_video(SRC_VIDEO_PATH, real_dir / "v04_real_group.mp4", "real")
    generate_video(SRC_VIDEO_PATH, real_dir / "v05_real_group.mp4", "real")
    
    # 2. Real Compressed videos (V06-V07)
    generate_video(SRC_VIDEO_PATH, real_dir / "v06_compressed.mp4", "real", compress=True)
    generate_video(SRC_VIDEO_PATH, real_dir / "v07_compressed.mp4", "real", compress=True)
    
    # 3. Deepfake Lip-Sync videos (V08-V10)
    print("\n--- Generating Deepfake Videos ---")
    generate_video(SRC_VIDEO_PATH, fake_dir / "v08_lipsync.mp4", "lip_sync")
    generate_video(SRC_VIDEO_PATH, fake_dir / "v09_lipsync.mp4", "lip_sync")
    generate_video(SRC_VIDEO_PATH, fake_dir / "v10_lipsync.mp4", "lip_sync")
    
    # 4. Deepfake Face-Swap videos (V11-V12)
    swap_face_1 = face_imgs[0]
    swap_face_2 = face_imgs[min(1, len(face_imgs) - 1)]
    generate_video(SRC_VIDEO_PATH, fake_dir / "v11_faceswap.mp4", "face_swap", swap_face_1)
    generate_video(SRC_VIDEO_PATH, fake_dir / "v12_faceswap.mp4", "face_swap", swap_face_2)
    
    print("\n--- Video Generation Completed! ---")

if __name__ == "__main__":
    main()
