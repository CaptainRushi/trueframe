"""
Diagnose which signal detectors are firing on each synthetic test image.
"""
import os
import sys
import json
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    _get_face_crops, _get_mtcnn_crops, _load_image,
    _signal_frequency_artifacts, _signal_block_artifacts,
    _signal_face_texture, _signal_blending_edges, _signal_noise_floor,
    _signal_face_gan_frequency, _signal_skin_tone_consistency,
    _signal_eye_region_artifacts, _signal_channel_decoupling,
    _signal_oversmoothed_skin, _signal_blur_similarity,
    _is_extreme_exposure, _has_skin_tone_pixels,
)

TEST_MEDIA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_media")

SIGNAL_FUNCS = {
    "frequency_artifacts": _signal_frequency_artifacts,
    "block_artifacts": _signal_block_artifacts,
    "face_texture": _signal_face_texture,
    "blending_edges": _signal_blending_edges,
    "noise_floor": _signal_noise_floor,
    "gan_frequency": _signal_face_gan_frequency,
    "skin_tone": _signal_skin_tone_consistency,
    "eye_artifacts": _signal_eye_region_artifacts,
    "channel_decoupling": _signal_channel_decoupling,
    "oversmoothed_skin": _signal_oversmoothed_skin,
    "blur_similarity": _signal_blur_similarity,
}

# Test images
test_paths = []
df_dir = os.path.join(TEST_MEDIA_DIR, "deepfake")
for fname in sorted(os.listdir(df_dir)):
    path = os.path.join(df_dir, fname)
    if os.path.isfile(path) and fname.endswith((".jpg", ".png")):
        test_paths.append((fname, path))

print(f"{'Image':45s} {'Faces':6s} {'Exposure':10s} {'FaceDet':8s}", end="")
for name in SIGNAL_FUNCS:
    print(f" {name[:14]:>14s}", end="")
print()

for fname, path in test_paths:
    img = _load_image(path)
    frames = [img] if img is not None else []
    
    mtcnn_crops = _get_mtcnn_crops(frames)
    has_faces = len(mtcnn_crops) >= 1
    crops = _get_face_crops(frames) if has_faces else mtcnn_crops
    
    extreme = _is_extreme_exposure(crops) if crops else False
    skin = _has_skin_tone_pixels(frames[0]) if frames else False
    
    print(f"{fname:45s} {str(has_faces):6s} {str(extreme):10s} {str(skin):8s}", end="")
    
    for sname, sfunc in SIGNAL_FUNCS.items():
        score = 0.0
        triggered = False
        try:
            if sname in ("frequency_artifacts", "block_artifacts"):
                score, triggered = sfunc(frames)
            elif sname in ("face_texture", "blending_edges", "noise_floor", "gan_frequency",
                          "skin_tone", "eye_artifacts", "channel_decoupling",
                          "oversmoothed_skin", "blur_similarity"):
                if crops:
                    score, triggered = sfunc(crops)
        except Exception:
            score = -1
        marker = "!" if triggered else "."
        print(f" {score:>13.4f}{marker}", end="")
    print()
