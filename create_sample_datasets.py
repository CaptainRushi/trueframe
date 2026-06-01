import os
import sys
import cv2
import json
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

print(f"Project root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}")

# Cascade file for face detection
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

if face_cascade.empty():
    print("Error: Could not load Haar cascade face detector!")
    sys.exit(1)

def get_face_box(frame):
    """Detects the largest face in the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    # Return the largest face by area
    largest = max(faces, key=lambda f: f[2] * f[3])
    return largest

def apply_face_swap(frame, face_img, box):
    """Blends a different face (from face_img) into the detected face box."""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return frame
    
    # Crop and resize source face image to match target face box
    src_face = cv2.resize(face_img, (w, h))
    
    # Create an elliptical mask to blend the face smoothly (simulating face swap boundary)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.45), int(h * 0.55)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (15, 15), 5)
    mask = np.expand_dims(mask, axis=-1)
    
    # Extract the face region and blend
    dst_frame = frame.copy()
    face_region = dst_frame[y:y+h, x:x+w]
    
    # Check bounds
    if face_region.shape[:2] == (h, w):
        blended = src_face * mask + face_region * (1.0 - mask)
        dst_frame[y:y+h, x:x+w] = blended.astype(np.uint8)
        
    return dst_frame

def apply_face_reenactment(frame, box):
    """Applies a subtle facial warp to simulate reenactment expression manipulation."""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return frame
    
    dst_frame = frame.copy()
    face_region = dst_frame[y:y+h, x:x+w]
    
    if face_region.shape[:2] == (h, w):
        # Create a wave/distortion warp using remap
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        # Distort the coordinates slightly in the center of the face
        cx, cy = w / 2.0, h / 2.0
        r_x = map_x - cx
        r_y = map_y - cy
        dist = np.sqrt(r_x**2 + r_y**2)
        
        # Apply a sinusoidal distortion to simulate mouth/expression warp
        factor = np.sin(dist / 10.0) * 3.0
        map_x = map_x + factor
        map_y = map_y + factor
        
        warped = cv2.remap(face_region, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dst_frame[y:y+h, x:x+w] = warped
        
    return dst_frame

def apply_neural_texture(frame, box):
    """Applies a stylized/bilateral neural-like texture filter to the face region."""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return frame
    
    dst_frame = frame.copy()
    face_region = dst_frame[y:y+h, x:x+w]
    
    if face_region.shape[:2] == (h, w):
        # Apply bilateral filter to smooth face details while keeping edges, creating a plastic/neural look
        stylized = cv2.bilateralFilter(face_region, 15, 75, 75)
        # Add a subtle periodic pattern overlay (high frequency texture artifacts)
        overlay = np.zeros_like(face_region)
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        grid = (np.sin(xx / 2.0) * np.cos(yy / 2.0) > 0.3).astype(np.float32) * 15.0
        overlay[:, :] = grid[:, :, np.newaxis]
        
        blended = cv2.addWeighted(stylized, 0.95, overlay.astype(np.uint8), 0.05, 0)
        dst_frame[y:y+h, x:x+w] = blended
        
    return dst_frame

def apply_lip_sync(frame, box):
    """Applies heavy pixelation/blur locally to the mouth region."""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return frame
    
    dst_frame = frame.copy()
    
    # Mouth is roughly bottom 1/3 of the face, centered horizontally
    my_start = y + int(h * 0.65)
    my_end = y + int(h * 0.95)
    mx_start = x + int(w * 0.22)
    mx_end = x + int(w * 0.78)
    
    mw = mx_end - mx_start
    mh = my_end - my_start
    
    if mw > 0 and mh > 0 and my_end <= frame.shape[0] and mx_end <= frame.shape[1]:
        mouth = dst_frame[my_start:my_end, mx_start:mx_end]
        # Pixelate
        small = cv2.resize(mouth, (max(1, mw // 8), max(1, mh // 8)), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (mw, mh), interpolation=cv2.INTER_NEAREST)
        # Smooth boundaries slightly
        pixelated_blurred = cv2.GaussianBlur(pixelated, (5, 5), 1.5)
        dst_frame[my_start:my_end, mx_start:mx_end] = pixelated_blurred
        
    return dst_frame

def apply_ai_generated(frame, box):
    """Applies GAN-style noise residuals and slight sharpness/color changes to simulate generated face."""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return frame
    
    dst_frame = frame.copy()
    face_region = dst_frame[y:y+h, x:x+w]
    
    if face_region.shape[:2] == (h, w):
        # Apply a slight sharpening filter and a custom GAN checkerboard artifact
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(face_region, -1, kernel)
        
        # Periodic checkerboard grid pattern
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        grid = (((xx // 4) % 2 == 0) & ((yy // 4) % 2 == 0)).astype(np.float32) * 8.0
        grid = np.expand_dims(grid, axis=-1)
        
        blended = (sharpened.astype(np.float32) + grid).clip(0, 255).astype(np.uint8)
        dst_frame[y:y+h, x:x+w] = blended
        
    return dst_frame

def generate_video(input_path, output_path, manipulation_type, face_img=None, duration=5.0):
    """Reads input video and writes the output video applying the specified manipulation."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Cannot open source video {input_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # We want exactly `duration` seconds of video
    max_frames = int(fps * duration)
    if total_frames > 0:
        max_frames = min(max_frames, total_frames)
        
    # Setup video writer (use mp4v codec for reliability on Windows)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    print(f"Generating video: {output_path} ({manipulation_type})")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        box = get_face_box(frame)
        
        if box is not None:
            if manipulation_type == "real":
                # Real video is untouched
                modified_frame = frame
            elif manipulation_type == "face_swap":
                modified_frame = apply_face_swap(frame, face_img, box)
            elif manipulation_type == "face_reenactment":
                modified_frame = apply_face_reenactment(frame, box)
            elif manipulation_type == "neural_texture":
                modified_frame = apply_neural_texture(frame, box)
            elif manipulation_type == "lip_sync":
                modified_frame = apply_lip_sync(frame, box)
            elif manipulation_type == "ai_generated":
                modified_frame = apply_ai_generated(frame, box)
            else:
                modified_frame = frame
        else:
            # Fallback if no face detected in a frame
            modified_frame = frame
            
        out.write(modified_frame)
        frame_count += 1
        
    cap.release()
    out.release()
    print(f"Done generating {output_path} ({frame_count} frames written)")
    return True

def main():
    # 1. Source Video
    src_video = PROJECT_ROOT / "176527-855920754_medium.mp4"
    if not src_video.exists():
        print(f"Error: Source video not found at {src_video}")
        sys.exit(1)
        
    # 2. Source Face image for swap
    face_img_path = PROJECT_ROOT / "test_assets" / "stylegan_tpdne" / "tpdne_01.jpg"
    if face_img_path.exists():
        face_img = cv2.imread(str(face_img_path))
    else:
        print(f"Warning: StyleGAN face image not found at {face_img_path}. Synthesizing a face shape.")
        # Create a dummy colored face image
        face_img = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.circle(face_img, (128, 128), 90, (180, 200, 255), -1) # face color
        cv2.circle(face_img, (90, 100), 10, (50, 50, 50), -1)     # eye 1
        cv2.circle(face_img, (166, 100), 10, (50, 50, 50), -1)    # eye 2
        cv2.ellipse(face_img, (128, 170), (40, 20), 0, 0, 180, (50, 50, 255), 3) # mouth
        
    # Define directories and outputs
    # A. FaceForensics
    ff_root = DATA_DIR / "FaceForensics"
    ff_real = ff_root / "original_sequences" / "youtube" / "c23" / "videos"
    ff_manip = ff_root / "manipulated_sequences"
    
    # B. DFDC
    dfdc_root = DATA_DIR / "DFDC" / "dfdc_train_part_0"
    
    # C. CelebDF
    celeb_root = DATA_DIR / "CelebDF"
    celeb_real = celeb_root / "Celeb-real"
    celeb_yt = celeb_root / "YouTube-real"
    celeb_syn = celeb_root / "Celeb-synthesis"
    
    # D. Custom Reels
    custom_root = DATA_DIR / "custom_reels"
    custom_real = custom_root / "real"
    custom_fake = custom_root / "fake"
    
    # Map task execution
    print("\n--- Starting Dataset Directory Generation ---\n")
    
    # 1. FaceForensics++ Samples
    print("[1/4] FaceForensics++ Samples...")
    generate_video(src_video, ff_real / "sample_real_ff.mp4", "real")
    generate_video(src_video, ff_manip / "Deepfakes" / "c23" / "videos" / "sample_deepfakes_ff.mp4", "face_swap", face_img)
    generate_video(src_video, ff_manip / "Face2Face" / "c23" / "videos" / "sample_face2face_ff.mp4", "face_reenactment")
    generate_video(src_video, ff_manip / "FaceSwap" / "c23" / "videos" / "sample_faceswap_ff.mp4", "face_swap", face_img)
    generate_video(src_video, ff_manip / "NeuralTextures" / "c23" / "videos" / "sample_neuraltextures_ff.mp4", "neural_texture")
    
    # 2. DFDC Samples
    print("\n[2/4] DFDC Samples...")
    real_dfdc_name = "sample_real_dfdc.mp4"
    fake_dfdc_name = "sample_fake_dfdc.mp4"
    generate_video(src_video, dfdc_root / real_dfdc_name, "real")
    generate_video(src_video, dfdc_root / fake_dfdc_name, "face_swap", face_img)
    
    # Create DFDC metadata.json
    metadata = {
        real_dfdc_name: {"label": "REAL", "split": "train"},
        fake_dfdc_name: {"label": "FAKE", "split": "train"}
    }
    with open(dfdc_root / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Created metadata.json at {dfdc_root / 'metadata.json'}")
    
    # 3. CelebDF Samples
    print("\n[3/4] CelebDF Samples...")
    generate_video(src_video, celeb_real / "sample_celeb_real.mp4", "real")
    generate_video(src_video, celeb_yt / "sample_youtube_real.mp4", "real")
    generate_video(src_video, celeb_syn / "sample_celeb_synthesis.mp4", "face_swap", face_img)
    
    # 4. Custom Reels Samples
    print("\n[4/4] Custom Reels Samples...")
    generate_video(src_video, custom_real / "sample_custom_real.mp4", "real")
    generate_video(src_video, custom_fake / "sample_custom_fake.mp4", "lip_sync")
    
    print("\n--- All Dataset Samples Generated Successfully! ---\n")

if __name__ == "__main__":
    main()
