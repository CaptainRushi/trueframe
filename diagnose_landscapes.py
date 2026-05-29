"""
Diagnose what face detection finds in landscape images.
"""
import sys, os
sys.path.insert(0, 'verified-stream/ai_service')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np

# Import the detector
from main import _get_detector, _haar_fallback, _has_skin_tone_pixels, _load_image

paths = [
    'verified-stream/ai_service/test_media/real/no_face/r41_landscape.jpg',
    'verified-stream/ai_service/test_media/real/no_face/r42_landscape.jpg',
    'verified-stream/ai_service/test_media/real/no_face/r43_landscape.jpg',
]

detector = _get_detector()

for path in paths:
    img = _load_image(path)
    if img is None:
        print(f"MISSING: {path}")
        continue
    h, w = img.shape[:2]
    img_area = h * w
    print(f"\n=== {os.path.basename(path)} === ({w}x{h}, area={img_area:,})")
    
    # Check skin tone
    has_skin = _has_skin_tone_pixels(img)
    print(f"  _has_skin_tone_pixels: {has_skin}")
    
    # Try primary detector
    face = detector(img)
    if face is not None:
        fh, fw = face.shape[:2]
        face_area = fh * fw
        ratio = face_area / img_area
        print(f"  Primary detector: FOUND {fw}x{fh} = {face_area} px, ratio={ratio:.4f} ({ratio*100:.2f}%)")
    else:
        print(f"  Primary detector: no face")
        # Try brightened
        bright = cv2.convertScaleAbs(img, alpha=1.4, beta=30)
        face = detector(bright)
        if face is not None:
            fh, fw = face.shape[:2]
            face_area = fh * fw
            ratio = face_area / img_area
            print(f"  Bright retry: FOUND {fw}x{fh} = {face_area} px, ratio={ratio:.4f} ({ratio*100:.2f}%)")
        else:
            print(f"  Bright retry: no face")
            # Try Haar
            face = _haar_fallback(img)
            if face is not None:
                fh, fw = face.shape[:2]
                face_area = fh * fw
                ratio = face_area / img_area
                print(f"  Haar fallback: FOUND {fw}x{fh} = {face_area} px, ratio={ratio:.4f} ({ratio*100:.2f}%)")
            else:
                print(f"  Haar fallback: no face")
                bright2 = cv2.convertScaleAbs(img, alpha=1.6, beta=50)
                face = _haar_fallback(bright2)
                if face is not None:
                    fh, fw = face.shape[:2]
                    face_area = fh * fw
                    ratio = face_area / img_area
                    print(f"  Bright Haar: FOUND {fw}x{fh} = {face_area} px, ratio={ratio:.4f} ({ratio*100:.2f}%)")
                else:
                    print(f"  Bright Haar: NO FACE ANYWHERE -> should take fail-closed path")
