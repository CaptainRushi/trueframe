"""
Trace the full score computation for specific test images.
"""
import os
import sys
import json
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import analyze, _run_signal_analysis, _get_face_crops, _get_mtcnn_crops, _load_image, _is_video

TEST_MEDIA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_media")

# Also directly test the signal v2 scores
from kaggle_eval.signals_v2 import run_all_v2_signals

for fname in ["crop_seam.png", "channel_decoupled_face.png", "heavy_jpeg_artifacts.png"]:
    path = os.path.join(TEST_MEDIA_DIR, "deepfake", fname)
    if not os.path.exists(path):
        continue
    
    print(f"\n{'='*60}")
    print(f"{fname}")
    print(f"{'='*60}")
    
    # Full analysis
    result = analyze(path)
    print(f"FINAL: score={result['final_score']:.4f} verdict={result['verdict']}")
    print(f"Signals: {[s for s in result['signals'] if s not in ('content_type_real_photo',)]}")
    
    # Raw signal scores
    img = _load_image(path)
    frames = [img] if img is not None else []
    video = _is_video(path)
    mtcnn_crops = _get_mtcnn_crops(frames)
    has_faces = len(mtcnn_crops) >= (2 if video else 1)
    crops = _get_face_crops(frames) if has_faces else mtcnn_crops
    
    signal_score, signals, raw = _run_signal_analysis(path, frames, crops, has_faces, video)
    
    print(f"\n  has_faces={has_faces} crops={len(crops)}")
    print(f"  signal_score from _run_signal_analysis: {signal_score:.4f}")
    
    # V2 signals
    if crops or frames:
        v2_test = run_all_v2_signals(crops if crops else frames, frames)
        print(f"  v2 raw: dct={v2_test.get('dct_artifacts',0):.4f} lap={v2_test.get('laplacian_pyramid',0):.4f} seam={v2_test.get('enhanced_seam',0):.4f} wavelet={v2_test.get('wavelet',0):.4f} hist={v2_test.get('color_histogram',0):.4f}")
