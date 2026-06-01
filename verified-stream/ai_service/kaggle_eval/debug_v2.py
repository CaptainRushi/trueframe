"""
Debug: check if v2 signals are running and what values they produce.
"""
import os
import sys
import json
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    _run_signal_analysis, _get_face_crops, _get_mtcnn_crops,
    _load_image, _is_video, _V2_SIGNALS_AVAILABLE,
)

print(f"V2 signals available: {_V2_SIGNALS_AVAILABLE}")

TEST_MEDIA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_media")

test_images = [
    "blending_seam_face.png",
    "channel_decoupled_face.png", 
    "crop_oversmoothed.png",
    "crop_seam.png",
    "seam_face.png",
]

for fname in test_images:
    path = os.path.join(TEST_MEDIA_DIR, "deepfake", fname)
    if not os.path.exists(path):
        print(f"\n{fname}: NOT FOUND")
        continue
    
    img = _load_image(path)
    frames = [img] if img is not None else []
    video = _is_video(path)
    
    mtcnn_crops = _get_mtcnn_crops(frames)
    has_faces = len(mtcnn_crops) >= (2 if video else 1)
    crops = _get_face_crops(frames) if has_faces else mtcnn_crops
    
    signal_score, signals, raw = _run_signal_analysis(
        path, frames, crops, has_faces, video
    )
    
    print(f"\n{fname}: has_faces={has_faces}")
    print(f"  v1 raw: frequency={raw.get('frequency',0):.4f} texture={raw.get('texture',0):.4f} edges={raw.get('edges',0):.4f} gan_freq={raw.get('gan_frequency',0):.4f} channel={raw.get('channel_decoupling',0):.4f}")
    print(f"  v2 raw: dct={raw.get('dct_artifacts',0):.4f} lap={raw.get('laplacian_pyramid',0):.4f} seam={raw.get('enhanced_seam',0):.4f} wavelet={raw.get('wavelet',0):.4f} hist={raw.get('color_histogram',0):.4f}")
    print(f"  signal_score={signal_score:.4f} signals={[s for s in signals if 'v2_' not in s][:8]}")
