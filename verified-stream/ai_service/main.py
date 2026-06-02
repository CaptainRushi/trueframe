"""
TrueFrame AI Service — Deepfake Detector (Images + Videos)
============================================================
Detection strategy: MODEL-FIRST with signal analysis fallback.

Priority 1 — LightFakeDetect ONNX model (MobileNetV2 + CBAM + GRU):
    Loads  ai_service/models/lightfakedetect.onnx  if present.
    Runs MTCNN → normalize → ONNX inference → P(fake).
    Trained on Celeb-DF / FF++ datasets.

Priority 2 — Signal analysis fallback (always available):
    Pure OpenCV + NumPy signal detectors.
    Works on any machine without training.
    Used when ONNX model is not yet available.

Score fusion when both run:
    final = 0.70 * model_score + 0.30 * signal_score

Verdict:
    final_score >= THRESHOLD_REJECT → REJECTED (deepfake detected)
    THRESHOLD_APPROVE <= final_score < THRESHOLD_REJECT → UNDER_REVIEW
    final_score < THRESHOLD_APPROVE → APPROVED (real content)

Usage:
    python main.py <file_path>

Output: JSON with scores and verdict.
"""

import sys
import os
import json
import time
import glob
import numpy as np
import cv2
from config import THRESHOLD_APPROVE, THRESHOLD_REJECT

# Enhanced signal detectors v2
try:
    from kaggle_eval.signals_v2 import run_all_v2_signals
    _V2_SIGNALS_AVAILABLE = True
except ImportError:
    _V2_SIGNALS_AVAILABLE = False

# HuggingFace model — lazy-imported so missing deps don't crash startup
try:
    from ai_core.models import HuggingFaceDeepfakeDetector as _HFDetectorClass
    _HF_AVAILABLE = True
except Exception:
    _HFDetectorClass = None
    _HF_AVAILABLE = False

# ─────────────────── CONFIG ──────────────────────────
MAX_FRAMES          = 20
FRAME_SIZE          = (224, 224)
IMAGENET_MEAN       = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD        = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Signal analysis weights (fallback path)
WEIGHT_MODEL        = 0.32
WEIGHT_ARTIFACT     = 0.18
WEIGHT_TEMPORAL     = 0.15
WEIGHT_EXPRESSION   = 0.20
WEIGHT_METADATA     = 0.07
WEIGHT_COMPRESSION  = 0.08


# ─────────────────── HELPERS ─────────────────────────

def _log(msg):
    print(msg, file=sys.stderr)


def _is_video(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.m4v', '.3gp'}


def _load_image(path):
    img = cv2.imread(path)
    return img


def _sample_video_frames(path, n=MAX_FRAMES):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    times = np.linspace(0.5, max(0.5, duration - 0.5), n)
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
    cap.release()
    return frames


# ─────────────────── ONNX MODEL INFERENCE ────────────

_ONNX_SESSIONS = []

def _load_all_onnx_models():
    """
    Discover and load all ONNX models from the models directory.
    Each model's input shape is auto-detected.
    Populates the global _ONNX_SESSIONS list with (session, input_info) tuples.
    """
    global _ONNX_SESSIONS
    if _ONNX_SESSIONS:
        return _ONNX_SESSIONS

    try:
        import onnxruntime as ort
    except ImportError:
        _log("[ONNX] onnxruntime not installed")
        return []

    from config import MODELS_DIR
    model_paths = sorted(glob.glob(os.path.join(MODELS_DIR, "*.onnx")))
    if not model_paths:
        _log("[ONNX] No .onnx models found in models/ directory")
        return []

    loaded = []
    for path in model_paths:
        try:
            sess = ort.InferenceSession(
                path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            inp = sess.get_inputs()[0]
            shape = inp.shape
            _log(f"[ONNX] Loaded: {os.path.basename(path)} — input={inp.name} shape={shape}")
            loaded.append((sess, {"name": inp.name, "shape": shape}))
        except Exception as e:
            _log(f"[ONNX] Skipped {os.path.basename(path)}: {e}")

    _ONNX_SESSIONS = loaded
    return loaded


def _run_onnx_inference_generic(sess, input_info, face_crops):
    """
    Run an ONNX model on face crops, auto-adapting to the model's expected input shape.

    Supports:
    - (N, 3, H, W) — per-image classifier (e.g., EfficientNet, ViT)
    - (N, T, 3, H, W) — temporal model (e.g., LightFakeDetect GRU)
    - (N, C, H, W) with C != 3 — feature models, etc.
    """
    try:
        shape = input_info["shape"]
        num_dims = len(shape)

        if num_dims == 4:
            if isinstance(shape[1], int) and shape[1] == 3:
                n, c, h, w = shape
                prepared = []
                for crop in face_crops:
                    resized = cv2.resize(crop, (int(w), int(h)))
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    norm = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    prepared.append(norm.transpose(2, 0, 1))
                batch = np.stack(prepared) if prepared else np.zeros((1, c, int(h), int(w)), dtype=np.float32)
            elif isinstance(shape[3], int) and shape[3] == 3:
                n, h, w, c = shape
                prepared = []
                for crop in face_crops:
                    resized = cv2.resize(crop, (int(w), int(h)))
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    prepared.append(rgb)
                batch = np.stack(prepared) if prepared else np.zeros((1, int(h), int(w), 3), dtype=np.float32)
            else:
                return None

            out = sess.run(None, {input_info["name"]: batch})
            probs = np.array(out[0]).flatten()
            if probs.size > 1:
                return float(np.clip(probs[1], 0.0, 1.0)) if probs.shape[0] >= 2 else float(np.clip(probs[0], 0.0, 1.0))
            return float(np.clip(probs[0], 0.0, 1.0))

        elif num_dims == 5:
            b, t, c, h, w = shape
            T = int(t) if isinstance(t, int) else min(len(face_crops), 10)
            prepared = []
            for crop in face_crops[:T]:
                resized = cv2.resize(crop, (int(w), int(h)))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                norm = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
                prepared.append(norm.transpose(2, 0, 1))
            while len(prepared) < T:
                prepared.append(prepared[-1] if prepared else np.zeros((int(c), int(h), int(w)), dtype=np.float32))
            seq = np.stack(prepared)[np.newaxis].astype(np.float32)
            out = sess.run(None, {input_info["name"]: seq})
            return float(np.clip(out[0].flatten()[0], 0.0, 1.0))

        return None

    except Exception as e:
        _log(f"[ONNX] Model inference error: {e}")
        return None


def _normalize_crop(face_bgr):
    """BGR uint8 → float32 (3, 224, 224) normalized to ImageNet stats."""
    face_bgr = cv2.resize(face_bgr, FRAME_SIZE)
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return normalized.transpose(2, 0, 1)   # (3, 224, 224)


_DETECTOR = None

# ─────────────────── HUGGINGFACE MODEL (fallback) ────

_HF_DETECTOR = None

def _get_hf_detector():
    """
    Lazy-initialise the HuggingFace deepfake detector singleton.
    Returns HuggingFaceDeepfakeDetector instance, or None if unavailable.
    """
    global _HF_DETECTOR
    if _HF_DETECTOR is not None:
        return _HF_DETECTOR
    if not _HF_AVAILABLE or _HFDetectorClass is None:
        _log("[HuggingFace] torch/transformers not installed — HF model unavailable")
        return None
    try:
        _log("[HuggingFace] Loading dima806/deepfake_vs_real_image_detection model...")
        _HF_DETECTOR = _HFDetectorClass("dima806/deepfake_vs_real_image_detection")
        if _HF_DETECTOR.model is None:
            _log("[HuggingFace] Model failed to load — will use signal-only fallback")
            _HF_DETECTOR = None
        else:
            _log("[HuggingFace] Model ready.")
    except Exception as e:
        _log(f"[HuggingFace] Detector init failed: {e}")
        _HF_DETECTOR = None
    return _HF_DETECTOR

# ── Second model: GAN/AI-generated image detector ─────────────────────────────
# dima806 = trained on FaceForensics++ face-swap deepfakes (good for videos/reels)
# prithivMLmods = ViT trained on GAN/SD/MJ-generated images (good for static fakes)
# Together they cover both categories of deepfakes on the platform.
_GAN_DETECTOR = None
_GAN_DETECTOR_FAILED = False   # set True after first failure to avoid repeated load attempts

def _get_gan_detector():
    """
    Lazy-initialise the GAN/AI-image-specific detector singleton.
    This is a SUPPLEMENTARY model — failure is gracefully handled.
    """
    global _GAN_DETECTOR, _GAN_DETECTOR_FAILED
    if _GAN_DETECTOR is not None:
        return _GAN_DETECTOR
    if _GAN_DETECTOR_FAILED:
        return None
    if not _HF_AVAILABLE or _HFDetectorClass is None:
        return None
    try:
        _log("[GAN-Detector] Loading prithivMLmods/Deep-Fake-Detector-v2-Model...")
        _GAN_DETECTOR = _HFDetectorClass("prithivMLmods/Deep-Fake-Detector-v2-Model")
        if _GAN_DETECTOR.model is None:
            _log("[GAN-Detector] Model failed to load — GAN detection will use signal fallback")
            _GAN_DETECTOR = None
            _GAN_DETECTOR_FAILED = True
        else:
            _log("[GAN-Detector] GAN detector ready.")
    except Exception as e:
        _log(f"[GAN-Detector] Init failed ({e}) — skipping GAN secondary model")
        _GAN_DETECTOR = None
        _GAN_DETECTOR_FAILED = True
    return _GAN_DETECTOR


def _build_detector():
    # ── MTCNN ──────────────────────────────────────────
    try:
        from facenet_pytorch import MTCNN
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mtcnn = MTCNN(
            image_size=224, margin=30, min_face_size=20,
            thresholds=[0.4, 0.5, 0.5],
            keep_all=False, post_process=False, device=device,
        )

        def _mtcnn(frame_bgr):
            from PIL import Image
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            try:
                face_t = mtcnn(pil)
                if face_t is not None:
                    face_np = face_t.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    return cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                # Retry with brightened image (helps dark/low-contrast portraits)
                arr   = np.array(pil, dtype=np.int32)
                bright = np.clip(arr + 40, 0, 255).astype(np.uint8)
                face_t = mtcnn(Image.fromarray(bright))
                if face_t is not None:
                    face_np = face_t.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    return cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                return None
            except Exception:
                return None

        _log("[Detector] MTCNN (facenet-pytorch) initialized successfully.")
        return _mtcnn
    except Exception:
        pass

    # ── MediaPipe ──────────────────────────────────────
    try:
        import mediapipe as mp
        try:
            det = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.4
            )
        except Exception:
            det = mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=0.4
            )

        def _mediapipe(frame_bgr):
            rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = det.process(rgb)
            if not results.detections:
                return None
            best = max(results.detections, key=lambda d: d.score[0])
            bbox = best.location_data.relative_bounding_box
            fh, fw = frame_bgr.shape[:2]
            px = bbox.width  * 0.15
            py = bbox.height * 0.15
            x1 = max(0, int((bbox.xmin - px) * fw))
            y1 = max(0, int((bbox.ymin - py) * fh))
            x2 = min(fw, int((bbox.xmin + bbox.width  + px) * fw))
            y2 = min(fh, int((bbox.ymin + bbox.height + py) * fh))
            if x2 - x1 < 32 or y2 - y1 < 32:
                return None
            return frame_bgr[y1:y2, x1:x2]

        _log("[Detector] MediaPipe initialized successfully.")
        return _mediapipe
    except Exception:
        pass

    # ── Haar cascade ───────────────────────────────────
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    
    def _haar(frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        for nb in [3, 2]:
            faces = cascade.detectMultiScale(gray, 1.05, nb, minSize=(32, 32))
            if len(faces) > 0:
                x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                pad = int(min(w, h) * 0.15)
                return frame_bgr[max(0, y-pad):min(frame_bgr.shape[0], y+h+pad),
                                 max(0, x-pad):min(frame_bgr.shape[1], x+w+pad)]
        return None

    _log("[Detector] Haar cascade initialized as fallback.")
    return _haar


def _get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = _build_detector()
    return _DETECTOR


# ── Haar cascade (permissive fallback, always available) ────────────────────
_HAAR_FACE_CASCADE = None

def _get_haar_cascade():
    global _HAAR_FACE_CASCADE
    if _HAAR_FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _HAAR_FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _HAAR_FACE_CASCADE


def _haar_fallback(frame_bgr):
    """Very permissive Haar detection used when primary detector misses faces.
    Tries histogram-equalized and plain grayscale with multiple neighbour settings.
    """
    cascade = _get_haar_cascade()
    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    for nb in [1, 2, 3]:
        for img in [gray_eq, gray]:
            faces = cascade.detectMultiScale(
                img, scaleFactor=1.05, minNeighbors=nb, minSize=(20, 20)
            )
            if len(faces) > 0:
                x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                pad = int(min(w, h) * 0.20)
                return frame_bgr[
                    max(0, y - pad): min(frame_bgr.shape[0], y + h + pad),
                    max(0, x - pad): min(frame_bgr.shape[1], x + w + pad),
                ]
    return None


def _enhance_frame_clahe(frame):
    """Apply CLAHE to improve face detection in poor lighting."""
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    except Exception:
        return frame


def _detect_face(frame):
    """Detect and return the largest face crop from a frame using cached detector."""
    detector = _get_detector()
    face = detector(frame)
    if face is None:
        face = detector(_enhance_frame_clahe(frame))
    return face


def _run_onnx_ensemble(frames):
    """
    Run all available ONNX models on frames and fuse their predictions.

    Returns (ensemble_score, model_count, signal) or (None, 0, '').
    """
    sessions = _load_all_onnx_models()
    if not sessions:
        return None, 0, ""

    # Get face crops
    face_crops = _get_face_crops(frames)
    if not face_crops:
        return None, 0, ""

    scores = []
    for sess, info in sessions:
        try:
            score = _run_onnx_inference_generic(sess, info, face_crops)
            if score is not None:
                scores.append(score)
        except Exception as e:
            _log(f"[ONNX] Model error: {e}")

    if not scores:
        return None, 0, ""

    mean_score = float(np.mean(scores))
    max_score = float(np.max(scores))
    # Use max if models disagree strongly (std > 0.3), else mean
    if len(scores) >= 2 and np.std(scores) > 0.3:
        fused = max_score
        _log(f"[ONNX-Ensemble] Models disagree: mean={mean_score:.4f}, max={max_score:.4f}, using max")
    else:
        fused = mean_score

    return fused, len(scores), "onnx_model_ensemble"


def _get_face_crops(frames, min_face_area_ratio=0.012):
    """Detect and crop faces from each frame. Uses primary detector (MTCNN),
    then CLAHE enhancement, brightened retry, then permissive Haar cascade.

    Uses multi-pass detection to maximize face capture rate:
    1. Primary detector on raw frame
    2. CLAHE-enhanced frame (better for poor lighting)
    3. Brightened frame
    4. Haar cascade fallback
    5. CLAHE + brightened Haar fallback

    min_face_area_ratio lowered to 0.012 (from 0.015) to catch smaller face
    detections typical of wide-angle portrait shots.
    """
    crops    = []
    detector = _get_detector()
    for frame in frames:
        img_area = frame.shape[0] * frame.shape[1] if frame is not None else 1
        face = detector(frame)
        if face is None:
            face = detector(_enhance_frame_clahe(frame))
        if face is None:
            bright = cv2.convertScaleAbs(frame, alpha=1.4, beta=30)
            face   = detector(bright)
        if face is None:
            face = detector(_enhance_frame_clahe(bright))
        if face is None:
            face = _haar_fallback(frame)
        if face is None:
            bright = cv2.convertScaleAbs(frame, alpha=1.6, beta=50)
            face   = _haar_fallback(bright)
        if face is None:
            face = _haar_fallback(_enhance_frame_clahe(frame))
        if face is not None and face.size > 0:
            face_area = face.shape[0] * face.shape[1]
            if face_area / img_area >= min_face_area_ratio:
                crops.append(cv2.resize(face, FRAME_SIZE))
            else:
                _log(f"[FaceDetect] Discarded tiny detection "
                     f"(face={face_area}px, image={img_area}px, "
                     f"ratio={face_area/img_area:.4f} < {min_face_area_ratio})")
    return crops


def _get_mtcnn_crops(frames):
    """Face detection for has_faces fail-closed determination.

    Strategy:
    1. Try MTCNN (neural face detector, no landscape false positives).
    2. If MTCNN fails AND frame has skin-tone pixels (probable portrait):
       try Haar fallback with a face-area-ratio filter.
       - Landscape Haar false positives are massive (43-49% of image) and are
         filtered out by the 0.38 upper bound.
       - Real portrait faces MTCNN misses are typically 10-35% of image.
       - r43 landscape has NO skin tone so Haar is never tried for it.
    3. If nothing found: has_faces = False → fail-closed path.
    """
    crops    = []
    detector = _get_detector()
    for frame in frames:
        img_area = frame.shape[0] * frame.shape[1] if frame is not None else 1
        face = detector(frame)
        if face is None:
            face = detector(_enhance_frame_clahe(frame))
        if face is None:
            bright = cv2.convertScaleAbs(frame, alpha=1.4, beta=30)
            face   = detector(bright)
        if face is None:
            face = detector(_enhance_frame_clahe(bright))
        if face is None and _has_skin_tone_pixels(frame):
            haar_face = _haar_fallback(frame)
            if haar_face is None:
                haar_face = _haar_fallback(_enhance_frame_clahe(frame))
            if haar_face is None:
                bright2 = cv2.convertScaleAbs(frame, alpha=1.6, beta=50)
                haar_face = _haar_fallback(bright2)
            if haar_face is not None and haar_face.size > 0:
                ratio = (haar_face.shape[0] * haar_face.shape[1]) / img_area
                if 0.015 <= ratio <= 0.42:
                    face = haar_face
                    _log(f"[FaceDetect] Skin-tone Haar fallback: "
                         f"ratio={ratio:.3f} ACCEPTED")
                else:
                    _log(f"[FaceDetect] Skin-tone Haar fallback: "
                         f"ratio={ratio:.3f} REJECTED (outside range)")
        if face is not None and face.size > 0:
            crops.append(cv2.resize(face, FRAME_SIZE))
    return crops



def _get_primary_face_track(frames):

    """Return a single face crop per frame (or None) for temporal analysis."""
    if not frames:
        return []
    detector = _get_detector()
    track    = []
    for frame in frames:
        face = detector(frame)
        if face is None:
            bright = cv2.convertScaleAbs(frame, alpha=1.4, beta=30)
            face   = detector(bright)
        if face is None:
            face = _haar_fallback(frame)
        track.append(face)
    return track


def _expression_rois(face_crop):
    h, w = face_crop.shape[:2]
    if h < 40 or w < 40:
        return None, None
    eye = face_crop[int(h * 0.22):int(h * 0.52), :]
    mouth = face_crop[int(h * 0.62):int(h * 0.90),
                      int(w * 0.18):int(w * 0.82)]
    if eye.size == 0 or mouth.size == 0:
        return None, None
    return eye, mouth


def _flow_magnitude(prev, curr):
    flow = cv2.calcOpticalFlowFarneback(
        prev, curr, None,
        pyr_scale=0.5, levels=2, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.1,
        flags=0,
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.mean(mag))


def _signal_expression_consistency(frames):
    if len(frames) < 4:
        return 0.0, False

    track = _get_primary_face_track(frames)
    mouth_moves = []
    eye_moves = []
    face_moves = []

    prev_face = None
    prev_eye = None
    prev_mouth = None

    for crop in track:
        if crop is None:
            prev_face = prev_eye = prev_mouth = None
            continue
        eye, mouth = _expression_rois(crop)
        if eye is None or mouth is None:
            prev_face = prev_eye = prev_mouth = None
            continue

        face_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)

        face_gray = cv2.resize(face_gray, (96, 96))
        eye_gray = cv2.resize(eye_gray, (64, 32))
        mouth_gray = cv2.resize(mouth_gray, (64, 32))

        if prev_face is not None:
            face_moves.append(_flow_magnitude(prev_face, face_gray))
            eye_moves.append(_flow_magnitude(prev_eye, eye_gray))
            mouth_moves.append(_flow_magnitude(prev_mouth, mouth_gray))

        prev_face, prev_eye, prev_mouth = face_gray, eye_gray, mouth_gray

    if len(mouth_moves) < 3 or len(face_moves) < 3:
        return 0.0, False

    face_mean = float(np.mean(face_moves))
    mouth_mean = float(np.mean(mouth_moves))
    eye_mean = float(np.mean(eye_moves)) if eye_moves else 0.0

    if face_mean < 1e-3 or mouth_mean < 1e-3:
        return 0.0, False

    mouth_ratio = mouth_mean / (face_mean + 1e-6)
    eye_ratio = eye_mean / (face_mean + 1e-6) if eye_mean > 0 else 0.0

    mouth_cv = float(np.std(mouth_moves)) / (mouth_mean + 1e-6)
    eye_cv = float(np.std(eye_moves)) / (eye_mean + 1e-6) if eye_mean > 0 else 0.0

    def _ratio_score(val, low, high):
        if val < low:
            return min(1.0, (low - val) / low)
        if val > high:
            return min(1.0, (val - high) / high)
        return 0.0

    def _cv_score(val, low, high):
        if val < low:
            return min(1.0, (low - val) / low)
        if val > high:
            return min(1.0, (val - high) / high)
        return 0.0

    mouth_ratio_score = _ratio_score(mouth_ratio, 0.6, 2.0)
    eye_ratio_score = _ratio_score(eye_ratio, 0.4, 1.6) if eye_mean > 0 else 0.0
    mouth_cv_score = _cv_score(mouth_cv, 0.15, 1.2)
    eye_cv_score = _cv_score(eye_cv, 0.12, 1.0) if eye_mean > 0 else 0.0

    score = (
        0.40 * mouth_ratio_score +
        0.25 * mouth_cv_score +
        0.20 * eye_ratio_score +
        0.15 * eye_cv_score
    )
    score = float(np.clip(score, 0.0, 1.0))
    triggered = score > 0.55 or mouth_ratio < 0.5 or mouth_ratio > 2.2 or \
        (eye_mean > 0 and (eye_ratio < 0.35 or eye_ratio > 1.8))

    return score, triggered


def _signal_frequency_artifacts(frames):
    if not frames:
        return 0.0, False
    scores = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        fft  = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shifted))
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        r_inner = int(min(h, w) * 0.40)
        r_outer = int(min(h, w) * 0.70)
        r_ref = int(min(h, w) * 0.20)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        hf_mask  = (dist >= r_inner) & (dist <= r_outer)
        ref_mask = dist <= r_ref
        hf_power  = magnitude[hf_mask].mean()
        ref_power = magnitude[ref_mask].mean()
        ratio = hf_power / (ref_power + 1e-6)
        scores.append(ratio)
    mean_ratio = float(np.mean(scores))
    score = min(1.0, max(0.0, (mean_ratio - 0.70) / 0.35))
    triggered = mean_ratio > 0.88
    return score, triggered


def _signal_block_artifacts(frames):
    if not frames:
        return 0.0, False
    scores = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        if w < 24 or h < 24:
            scores.append(0.0)
            continue
        # Detect 8x8 JPEG block boundary seams:
        # Compare pixel differences ACROSS block boundaries vs WITHIN blocks
        # At each 8-pixel boundary, compute |pixel(boundary) - pixel(boundary-1)|
        # vs |pixel(boundary+4) - pixel(boundary+3)| for interior positions
        boundary_diffs = []
        interior_diffs = []
        for y in range(0, h, 8):
            for x in range(8, w - 8, 8):
                # Across boundary at column x
                bd = abs(float(gray[y, x]) - float(gray[y, x - 1]))
                boundary_diffs.append(bd)
                # Interior (within block) at column x+4
                id_val = abs(float(gray[y, x + 4]) - float(gray[y, x + 3]))
                interior_diffs.append(id_val)
        if not boundary_diffs or not interior_diffs:
            scores.append(0.0)
            continue
        bd_mean = float(np.mean(boundary_diffs))
        id_mean = float(np.mean(interior_diffs))
        ratio = bd_mean / (id_mean + 1e-6)
        scores.append(max(0.0, ratio - 1.0))
    if not scores:
        return 0.0, False
    mean_ratio = float(np.mean(scores))
    score = min(1.0, max(0.0, mean_ratio * 5.0))
    triggered = mean_ratio > 0.30
    return score, triggered


def _signal_face_texture(crops):
    if not crops:
        return 0.0, False
    variances = []
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap  = cv2.Laplacian(gray, cv2.CV_64F)
        variances.append(float(np.var(lap)))
    mean_var = float(np.mean(variances))
    std_var  = float(np.std(variances))
    too_smooth   = mean_var < 65.0
    too_sharp    = mean_var > 15000.0
    unstable     = std_var / (mean_var + 1.0) > 1.8
    triggered = too_smooth or too_sharp or unstable
    if too_smooth:
        score = min(1.0, 65.0 / (mean_var + 1.0))
    elif too_sharp:
        score = min(1.0, (mean_var - 15000.0) / 10000.0)
    elif unstable:
        score = min(1.0, (std_var / (mean_var + 1.0) - 1.6) / 2.0)
    else:
        score = 0.0
    return score, triggered


def _signal_blending_edges(crops):
    """Detect face-swap blending seams: abrupt high-freq discontinuities at face border."""
    if not crops:
        return 0.0, False
    edge_ratios = []
    variance_scores = []
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sobel = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        )
        h, w = sobel.shape
        bw = max(8, int(min(h, w) * 0.12))
        border = np.concatenate([
            sobel[:bw, :].ravel(), sobel[-bw:, :].ravel(),
            sobel[:, :bw].ravel(), sobel[:, -bw:].ravel(),
        ])
        interior = sobel[bw:-bw, bw:-bw].ravel()
        if interior.size == 0:
            continue
        ratio = float(np.mean(border)) / (float(np.mean(interior)) + 1e-6)
        edge_ratios.append(ratio)
        # Also check abrupt variance difference (seam creates local variance spike)
        border_var   = float(np.var(border))
        interior_var = float(np.var(interior))
        variance_scores.append(border_var / (interior_var + 1e-6))
    if not edge_ratios:
        return 0.0, False
    mean_ratio = float(np.mean(edge_ratios))
    mean_var_ratio = float(np.mean(variance_scores)) if variance_scores else 1.0
    deviation = abs(mean_ratio - 1.0)
    var_deviation = max(0.0, mean_var_ratio - 1.0)
    triggered = deviation > 0.15 or var_deviation > 0.40
    score = min(1.0, deviation / 0.5 * 0.6 + min(1.0, var_deviation / 1.5) * 0.4)
    return score, triggered


def _signal_noise_floor(crops):
    """Detect unnaturally smooth (GAN) or inconsistently noisy faces."""
    if not crops:
        return 0.0, False
    noise_levels = []
    for crop in crops:
        gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_levels.append(float(np.mean(np.abs(gray - blurred))))
    mean_noise = float(np.mean(noise_levels))
    std_noise  = float(np.std(noise_levels))
    too_clean    = mean_noise < 0.8
    inconsistent = std_noise / (mean_noise + 1e-6) > 1.3
    triggered = too_clean or inconsistent
    if too_clean:
        score = min(1.0, 0.8 / (mean_noise + 0.05))
    elif inconsistent:
        score = min(1.0, (std_noise / (mean_noise + 1e-6) - 1.3) / 2.0)
    else:
        score = 0.0
    return score, triggered


def _signal_face_gan_frequency(crops):
    """Detect GAN spectral fingerprints.

    Two modes:
    - HIGH ratio (> 0.58): GAN spectral artifacts / excess high-frequency noise
    - LOW  ratio (< 0.38): Oversmoothed GAN face with deficient high-frequency content
    """
    if not crops:
        return 0.0, False
    scores = []
    raw_ratios = []
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        fft = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shifted))
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        r_hf_in  = int(min(h, w) * 0.50)
        r_hf_out = int(min(h, w) * 0.80)
        r_lf     = int(min(h, w) * 0.25)
        hf_mask = (dist >= r_hf_in) & (dist <= r_hf_out)
        lf_mask = dist <= r_lf
        hf_power = magnitude[hf_mask].mean() if hf_mask.any() else 0.0
        lf_power = magnitude[lf_mask].mean() if lf_mask.any() else 1.0
        ratio = hf_power / (lf_power + 1e-6)
        raw_ratios.append(ratio)
    mean_ratio = float(np.mean(raw_ratios))
    high_hf_score = min(1.0, max(0.0, (mean_ratio - 0.50) / 0.40))
    low_hf_score  = min(1.0, max(0.0, (0.40 - mean_ratio) / 0.20))
    score     = max(high_hf_score, low_hf_score)
    triggered = mean_ratio > 0.54 or mean_ratio < 0.40
    return score, triggered


def _signal_skin_tone_consistency(crops):
    if len(crops) < 3:
        # For single images: GAN faces may have unnaturally UNIFORM skin
        # But single-crop hue std is unreliable - skip
        return 0.0, False
    skin_hues = []
    skin_sats = []
    for crop in crops:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        skin_mask = (
            ((H >= 0) & (H <= 25)) | ((H >= 165) & (H <= 180))
        ) & (S >= 40) & (V >= 50)
        if skin_mask.sum() < 100:
            continue
        skin_hues.append(float(H[skin_mask].mean()))
        skin_sats.append(float(S[skin_mask].mean()))
    if len(skin_hues) < 3:
        return 0.0, False
    hue_std = float(np.std(skin_hues))
    sat_std = float(np.std(skin_sats))
    hue_score = min(1.0, max(0.0, (hue_std - 2.5) / 8.0))
    sat_score = min(1.0, max(0.0, (sat_std - 10.0) / 20.0))
    score = 0.6 * hue_score + 0.4 * sat_score
    triggered = hue_std > 4.0 or sat_std > 16.0
    return float(score), triggered


def _signal_eye_region_artifacts(crops):
    if not crops:
        return 0.0, False
    channel_corrs = []
    edge_entropies = []
    for crop in crops:
        h = crop.shape[0]
        eye = crop[int(h * 0.25): int(h * 0.52), :]
        if eye.size == 0:
            continue
        b = eye[:, :, 0].astype(np.float64).ravel()
        g = eye[:, :, 1].astype(np.float64).ravel()
        r = eye[:, :, 2].astype(np.float64).ravel()
        if np.std(b) < 1e-6 or np.std(g) < 1e-6:
            continue
        corr_rg = float(np.corrcoef(r, g)[0, 1])
        corr_rb = float(np.corrcoef(r, b)[0, 1])
        channel_corrs.append(min(corr_rg, corr_rb))
        gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sobel = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        )
        hist, _ = np.histogram(sobel.ravel(), bins=32, range=(0, 150))
        hist = (hist + 1e-9) / (hist.sum() + 1e-9)
        entropy = float(-np.sum(hist * np.log2(hist)))
        edge_entropies.append(entropy)
    if not channel_corrs:
        return 0.0, False
    mean_corr = float(np.mean(channel_corrs))
    corr_score = min(1.0, max(0.0, (0.82 - mean_corr) / 0.30))
    mean_entropy = float(np.mean(edge_entropies)) if edge_entropies else 4.0
    ent_score = min(1.0, max(0.0, abs(mean_entropy - 4.0) / 1.5))
    score = 0.65 * corr_score + 0.35 * ent_score
    triggered = mean_corr < 0.75 or abs(mean_entropy - 4.0) > 1.1
    return float(score), triggered


def _signal_channel_decoupling(crops):
    """Detect decoupled RGB channels -- hallmark of GAN/face-swap artefacts."""
    if not crops:
        return 0.0, False
    corr_scores = []
    mean_diffs  = []
    for crop in crops:
        b = crop[:, :, 0].astype(np.float64).ravel()
        g = crop[:, :, 1].astype(np.float64).ravel()
        r = crop[:, :, 2].astype(np.float64).ravel()
        if np.std(b) < 1e-6 or np.std(g) < 1e-6 or np.std(r) < 1e-6:
            continue
        rg = float(np.corrcoef(r, g)[0, 1])
        rb = float(np.corrcoef(r, b)[0, 1])
        gb = float(np.corrcoef(g, b)[0, 1])
        corr_scores.append(min(rg, rb, gb))
        mean_diffs.append(
            abs(float(r.mean()) - float(g.mean())) +
            abs(float(r.mean()) - float(b.mean()))
        )
    if not corr_scores:
        return 0.0, False
    mean_min_corr = float(np.mean(corr_scores))
    mean_ch_diff  = float(np.mean(mean_diffs)) if mean_diffs else 0.0
    corr_score = min(1.0, max(0.0, (0.90 - mean_min_corr) / 0.25))
    diff_score = min(1.0, mean_ch_diff / 50.0)
    score      = 0.7 * corr_score + 0.3 * diff_score
    triggered  = mean_min_corr < 0.82 or mean_ch_diff > 25.0
    return score, triggered


def _signal_oversmoothed_skin(crops):
    """Detect GAN oversmoothed faces via patch-level variance uniformity.

    GAN-generated oversmoothed faces have unnaturally uniform low variance
    across ALL face patches. Real faces have heterogeneous texture (skin pores,
    shadows, hair boundaries) producing a mix of low- and high-variance patches.
    """
    if not crops:
        return 0.0, False
    patch_var_means = []
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        ph, pw = max(1, h // 4), max(1, w // 4)
        if ph < 8 or pw < 8:
            continue
        patch_vars = []
        for py in range(4):
            for px in range(4):
                patch = gray[py * ph:(py + 1) * ph, px * pw:(px + 1) * pw]
                if patch.size > 0:
                    patch_vars.append(float(np.var(patch)))
        if len(patch_vars) >= 8:
            patch_var_means.append(float(np.mean(patch_vars)))
    if not patch_var_means:
        return 0.0, False
    mean_pv   = float(np.mean(patch_var_means))
    too_uniform = mean_pv < 100.0
    triggered   = too_uniform
    score       = min(1.0, 100.0 / (mean_pv + 1.0)) if too_uniform else 0.0
    return score, triggered


def _signal_blur_similarity(crops):
    """Detect already-blurred (GAN oversmoothed) faces using brightness-normalised MSE.

    Applies an additional Gaussian blur to each crop and measures how much the image
    changes. An oversmoothed GAN face is already at its frequency limit, so further
    blurring barely changes it → very low normalised MSE → triggered.
    A real face — even a dark one — retains enough skin micro-texture that further
    blurring produces a meaningfully larger difference.

    This metric is brightness-invariant (divides MSE by mean_brightness^2) so it
    remains reliable for dark or dim face crops without producing false positives.
    """
    if not crops:
        return 0.0, False
    norm_mse_values = []
    for crop in crops:
        gray       = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean_b     = max(5.0, gray.mean())          # avoid /0
        blurred    = cv2.GaussianBlur(gray, (21, 21), 5)
        mse        = float(np.mean((gray - blurred) ** 2))
        norm_mse   = mse / (mean_b ** 2) * 10_000   # scale to 0-~1000
        norm_mse_values.append(norm_mse)
    if not norm_mse_values:
        return 0.0, False
    mean_norm_mse = float(np.mean(norm_mse_values))
    too_smooth = mean_norm_mse < 20.0
    triggered  = too_smooth
    score      = min(1.0, 20.0 / (mean_norm_mse + 0.5)) if too_smooth else 0.0
    return score, triggered


def _signal_temporal_flicker(frames):
    if len(frames) < 4:
        return 0.0, False
    brightnesses = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() for f in frames]
    diffs = np.abs(np.diff(brightnesses))
    mean_diff = float(np.mean(diffs))
    max_diff  = float(np.max(diffs))
    score = min(1.0, (mean_diff / 6.0) * 0.4 + (max_diff / 30.0) * 0.6)
    triggered = mean_diff > 5.0 or max_diff > 20.0
    return score, triggered


def _signal_color_consistency(frames):
    if len(frames) < 4:
        return 0.0, False
    hue_means = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue_means.append(float(hsv[:, :, 0].mean()))
    diffs = np.abs(np.diff(hue_means))
    mean_diff = float(np.mean(diffs))
    max_diff  = float(np.max(diffs))
    score = min(1.0, (mean_diff / 8.0) * 0.5 + (max_diff / 40.0) * 0.5)
    triggered = mean_diff > 5.0 or max_diff > 25.0
    return score, triggered


def _signal_metadata(path):
    try:
        size = os.path.getsize(path)
        if _is_video(path) and size < 50_000:
            return 0.3, True
        if not _is_video(path) and size < 2_000:
            return 0.2, True
    except Exception:
        pass
    return 0.0, False


def _signal_ela_analysis(crops):
    """
    Error Level Analysis (ELA) — detects JPEG re-compression inconsistencies.

    Deepfakes are re-encoded multiple times: original → GAN → export.
    Each re-compression changes the JPEG Error Level Analysis signature.
    Real single-compressed photos have consistent ELA across the image.
    Fake/manipulated images show different ELA levels in face vs background.

    Higher score = more suspicious.
    """
    if not crops:
        return 0.0, False

    scores = []
    for crop in crops:
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Re-compress at known quality
            _, encoded = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
            if decoded is None:
                continue

            # ELA difference map
            ela_map = cv2.absdiff(gray.astype(np.float32), decoded.astype(np.float32))

            h, w = ela_map.shape
            # Split face into face-center and border regions
            cy, cx = h // 2, w // 2
            center_mask = np.zeros((h, w), dtype=bool)
            center_mask[cy - h // 4:cy + h // 4, cx - w // 4:cx + w // 4] = True
            border_mask = ~center_mask

            center_ela = ela_map[center_mask].mean()
            border_ela = ela_map[border_mask].mean()
            if center_ela < 0.1:
                continue

            ela_ratio = border_ela / (center_ela + 1e-6)
            ela_std = float(np.std(ela_map))

            # Real photos: center and border ELA are similar (ratio ~0.8-1.2)
            # Deepfakes: center (face) has different compression history than border → ratio deviates
            deviation = abs(ela_ratio - 1.0)
            score = min(1.0, deviation * 2.0)
            # Bonus for very high ELA variance (multiple compression cycles)
            if ela_std > 8.0:
                score = min(1.0, score + 0.15)
            scores.append(score)
        except Exception:
            continue

    if not scores:
        return 0.0, False

    mean_score = float(np.mean(scores))
    triggered = mean_score > 0.30
    return mean_score, triggered


def _signal_chromatic_aberration(crops):
    """
    Detect chromatic aberration inconsistencies — a lens artifact.

    Real camera lenses produce slight chromatic aberration (color fringing
    at high-contrast edges). GAN-generated faces have either:
    - No chromatic aberration (too perfect)
    - Unnatural color fringing patterns

    Analyzes color channel misalignment at edge boundaries.
    """
    if not crops:
        return 0.0, False

    scores = []
    for crop in crops:
        try:
            if crop.shape[2] < 3:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            if edges.sum() < 100:
                continue

            b, g, r = cv2.split(crop.astype(np.float32))
            # Measure color channel shift at edge pixels
            edge_y, edge_x = np.where(edges > 0)
            rg_diff = np.abs(r[edge_y, edge_x] - g[edge_y, edge_x])
            gb_diff = np.abs(g[edge_y, edge_x] - b[edge_y, edge_x])
            rb_diff = np.abs(r[edge_y, edge_x] - b[edge_y, edge_x])

            # Real lenses: moderate channel differences at edges (natural CA)
            # GANs: very low (no CA) or very high (unrealistic CA)
            mean_rg = float(np.mean(rg_diff))
            mean_gb = float(np.mean(gb_diff))
            mean_rb = float(np.mean(rb_diff))

            # Too little CA (GAN oversmoothed edges)
            low_ca = mean_rg < 1.5 and mean_gb < 1.5
            # Too much or inconsistent CA
            high_ca = mean_rb > 20.0

            if low_ca and high_ca:
                scores.append(0.6)
            elif low_ca:
                scores.append(min(1.0, (1.5 - mean_rg) / 1.5))
            elif high_ca:
                scores.append(min(1.0, (mean_rb - 20.0) / 20.0))
            else:
                scores.append(0.0)
        except Exception:
            continue

    if not scores:
        return 0.0, False

    mean_score = float(np.mean(scores))
    triggered = mean_score > 0.35
    return mean_score, triggered


def _signal_noise_inconsistency(crops):
    """
    Detect noise pattern inconsistency.

    Real camera photos have characteristic sensor noise patterns.
    GAN-generated images lack realistic sensor noise, or have
    uniform synthetic noise across the entire image.

    Measures: local noise variance across patches.
    Real photos: noise varies naturally (shadows = more noise, bright = less).
    GANs: noise is unnaturally uniform or absent.
    """
    if not crops:
        return 0.0, False

    scores = []
    for crop in crops:
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
            h, w = gray.shape
            # Extract noise by subtracting Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
            noise = gray - blurred

            # Divide into 8x8 grid and compute noise variance per patch
            ph, pw = h // 8, w // 8
            patch_noise_vars = []
            for py in range(8):
                for px in range(8):
                    y1, y2 = py * ph, (py + 1) * ph
                    x1, x2 = px * pw, (px + 1) * pw
                    patch = noise[y1:y2, x1:x2]
                    if patch.size > 0:
                        patch_noise_vars.append(float(np.var(patch)))

            if len(patch_noise_vars) < 16:
                continue

            # Real photos: noise variance varies significantly across patches
            # GANs: noise is uniform (low CV) or absent (very low mean)
            noise_mean = float(np.mean(patch_noise_vars))
            noise_cv = float(np.std(patch_noise_vars)) / (noise_mean + 1e-6)

            # Too clean (GAN): very low noise everywhere
            too_clean = noise_mean < 0.3
            # Too uniform (GAN): all patches have same noise level
            too_uniform = noise_cv < 0.3 and noise_mean < 3.0

            if too_clean and too_uniform:
                scores.append(0.7)
            elif too_clean:
                scores.append(min(1.0, (0.5 - noise_mean) / 0.5))
            elif too_uniform:
                scores.append(min(1.0, (0.4 - noise_cv) / 0.4))
            else:
                scores.append(0.0)
        except Exception:
            continue

    if not scores:
        return 0.0, False

    mean_score = float(np.mean(scores))
    triggered = mean_score > 0.30
    return mean_score, triggered


# ─────────────────── IMAGE QUALITY HELPERS ───────────

def _is_extreme_exposure(crops):
    """Returns True if face crops are severely over- or under-exposed.

    Overexposed (mean > 210) and underexposed (mean < 75) images fool the
    GAN/texture detectors because blown-out / dark faces look like smooth,
    noiseless GAN artefacts to our signal functions.
    Raised underexposure threshold from 45 to 75 to handle dark portrait crops.
    """
    if not crops:
        return False
    means = [cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32).mean() for c in crops]
    m = float(np.mean(means))
    return m > 210 or m < 75


def _has_skin_tone_pixels(frame):
    """Returns True if the frame contains a meaningful proportion of human skin tones.

    Used to distinguish a real portrait (face missed by MTCNN) from a landscape
    or object photo in the no-face fallback path. Uses a broad HSV range to
    cover diverse skin tones (light, dark, warm, cool).
    """
    if frame is None or frame.size == 0:
        return False
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Broad skin-tone coverage:
    # - Normal skin tones: H 0-25, S >= 10, V >= 40
    # - Dark/neutral skin: H 0-30, S >= 8, V >= 25
    # - Very light skin (low saturation): H 0-20, S 5-40, V >= 150
    # - Wrap-around red skin tones: H 160-179
    skin_mask = (
        (((H >= 0) & (H <= 30)) | ((H >= 155) & (H <= 179)))
        & (S >= 8) & (V >= 25)
    )
    skin_ratio = float(skin_mask.sum()) / (frame.shape[0] * frame.shape[1])
    return skin_ratio > 0.03   # at least 3% skin-tone pixels (lowered from 5%)


# ─────────────────── SIGNAL ANALYSIS ENGINE ──────────

def _run_signal_analysis(file_path, frames, crops, has_faces, video):
    """Run all signal detectors. Returns (final_score, signals, raw_scores)."""
    signals = []
    raw = {}

    freq_score, freq_trig = _signal_frequency_artifacts(frames)
    raw["frequency"] = freq_score
    if freq_trig:
        signals.append("high_frequency_artifacts")

    comp_score, comp_trig = _signal_block_artifacts(frames)
    raw["compression"] = comp_score
    if comp_trig:
        signals.append("reencoding_block_artifacts")

    if has_faces:
        # Check for extreme exposure — lighting artefacts mimic GAN signals,
        # so suppress exposure-sensitive detectors for very bright/dark faces.
        extreme_lighting = _is_extreme_exposure(crops)

        gray_means  = [cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32).mean() for c in crops]
        extreme_bright = float(np.mean(gray_means)) > 210

        tex_score, tex_trig = _signal_face_texture(crops)
        raw["texture"] = tex_score
        if tex_trig and not extreme_lighting:
            signals.append("unnatural_face_texture")

        edge_score, edge_trig = _signal_blending_edges(crops)
        raw["edges"] = edge_score
        if edge_trig:
            signals.append("face_blending_seam")

        noise_score, noise_trig = _signal_noise_floor(crops)
        raw["noise"] = noise_score
        if noise_trig and not extreme_lighting:
            signals.append("abnormal_noise_pattern")

        gan_freq_score, gan_freq_trig = _signal_face_gan_frequency(crops)
        raw["gan_frequency"] = gan_freq_score
        if gan_freq_trig and not extreme_lighting:
            signals.append("gan_spectral_fingerprint")

        skin_score, skin_trig = _signal_skin_tone_consistency(crops)
        raw["skin_tone"] = skin_score
        if skin_trig:
            signals.append("skin_tone_instability")

        eye_score, eye_trig = _signal_eye_region_artifacts(crops)
        raw["eye_artifacts"] = eye_score
        if eye_trig and not extreme_lighting:
            signals.append("eye_region_gan_artifact")

        channel_score, channel_trig = _signal_channel_decoupling(crops)
        raw["channel_decoupling"] = channel_score
        if channel_trig and not extreme_lighting:
            signals.append("color_channel_decoupled")

        oversmooth_score, oversmooth_trig = _signal_oversmoothed_skin(crops)
        raw["oversmoothed"] = oversmooth_score
        if oversmooth_trig and not extreme_lighting:
            signals.append("oversmoothed_skin_detected")

        blur_score, blur_trig = _signal_blur_similarity(crops)
        raw["blur_similarity"] = blur_score
        if blur_trig and not extreme_bright:
            signals.append("oversmoothed_blur_artifact")

        # ── New Enhanced Signals ────────────────────────────────
        ela_score, ela_trig = _signal_ela_analysis(crops)
        raw["ela"] = ela_score
        if ela_trig and not extreme_lighting:
            signals.append("ela_recompression_artifact")

        chroma_score, chroma_trig = _signal_chromatic_aberration(crops)
        raw["chromatic_aberration"] = chroma_score
        if chroma_trig:
            signals.append("chromatic_aberration_anomaly")

        noise_inc_score, noise_inc_trig = _signal_noise_inconsistency(crops)
        raw["noise_inconsistency"] = noise_inc_score
        if noise_inc_trig and not extreme_lighting:
            signals.append("noise_pattern_inconsistency")

        # ── V2 Enhanced Signal Detectors ────────────────────────────
        if _V2_SIGNALS_AVAILABLE:
            v2_raw = run_all_v2_signals(crops, frames)
            for k, v in v2_raw.items():
                raw[k] = v
            if raw.get("enhanced_seam", 0) > 0.40:
                signals.append("enhanced_blending_seam")
            if raw.get("dct_artifacts", 0) > 0.45:
                signals.append("dct_coefficient_anomaly")
            if raw.get("wavelet", 0) > 0.40:
                signals.append("wavelet_texture_artifact")
            if raw.get("laplacian_pyramid", 0) > 0.45:
                signals.append("laplacian_texture_artifact")
            if raw.get("color_histogram", 0) > 0.45:
                signals.append("color_histogram_anomaly")
    else:
        signals.append("no_clear_faces_detected")
        for k in ["texture", "edges", "noise", "gan_frequency",
                  "skin_tone", "eye_artifacts", "channel_decoupling", "oversmoothed",
                  "blur_similarity", "ela", "chromatic_aberration", "noise_inconsistency"]:
            raw[k] = 0.0

        if _V2_SIGNALS_AVAILABLE and frames:
            v2_raw = run_all_v2_signals(frames, frames)
            raw["dct_artifacts"] = v2_raw.get("dct_artifacts", 0.0)
            raw["wavelet"] = v2_raw.get("wavelet", 0.0)
            raw["laplacian_pyramid"] = v2_raw.get("laplacian_pyramid", 0.0)
            raw["enhanced_seam"] = v2_raw.get("enhanced_seam", 0.0)
            raw["color_histogram"] = v2_raw.get("color_histogram", 0.0)
            if raw.get("enhanced_seam", 0) > 0.45:
                signals.append("enhanced_blending_seam")
        else:
            for k in ["dct_artifacts", "laplacian_pyramid", "enhanced_seam", "wavelet", "color_histogram"]:
                raw[k] = 0.0

    if video:
        flicker_score, flicker_trig = _signal_temporal_flicker(frames)
        raw["temporal"] = flicker_score
        if flicker_trig:
            signals.append("temporal_face_distortion")

        color_score, color_trig = _signal_color_consistency(frames)
        raw["color"] = color_score
        if color_trig:
            signals.append("color_temperature_inconsistency")
    else:
        raw["temporal"] = 0.0
        raw["color"]    = 0.0

    if video and has_faces:
        expression_score, expression_trig = _signal_expression_consistency(frames)
        raw["expression"] = expression_score
        if expression_trig:
            signals.append("facial_expression_inconsistency")
    else:
        raw["expression"] = 0.0

    meta_score, meta_trig = _signal_metadata(file_path)
    if meta_trig:
        signals.append("suspicious_metadata_integrity")

    # Fail-Closed Logic:
    # For VIDEOS: no face detected = fail_closed (score=1.0) — we can't verify authenticity
    # For IMAGES: no face detected = send to UNDER_REVIEW rather than hard REJECT.
    #             The signal analysis on the full frame determines the final score.
    if not has_faces:
        if video:
            # Videos MUST have faces to be verified — strict fail-closed
            signals.append("fail_closed")
            return 1.0, signals, raw
        else:
            # Image with no face: use skin-tone presence to distinguish
            # a missed portrait (real user) from a landscape/object image.
            # Send to UNDER_REVIEW territory (0.35-0.50 base) rather than hard REJECT
            # so legitimate no-face content isn't blocked outright.
            frame_signal_score = (
                raw.get("frequency", 0.0) * 0.20 +
                raw.get("compression", 0.0) * 0.15 +
                0.0  # v2 signals excluded (baseline noise on JPEG real photos)
            )
            if frames and _has_skin_tone_pixels(frames[0]):
                no_face_base = 0.35
                signals.append("face_detection_missed_portrait")
            else:
                no_face_base = 0.45
            combined = float(np.clip(no_face_base + frame_signal_score * 0.40, 0.0, 1.0))
            return combined, signals, raw

    # New signal scores
    ela_score = raw.get("ela", 0.0)
    chroma_score = raw.get("chromatic_aberration", 0.0)
    noise_inc_score = raw.get("noise_inconsistency", 0.0)

    if video:
        v1_raw = (
            raw["frequency"]         * 0.05 +
            raw["texture"]           * 0.08 +
            raw["edges"]             * 0.08 +
            raw["noise"]             * 0.06 +
            raw["color"]             * 0.05 +
            raw["gan_frequency"]     * 0.18 +
            raw["skin_tone"]         * 0.10 +
            raw["eye_artifacts"]     * 0.12 +
            raw["channel_decoupling"]* 0.10 +
            ela_score                * 0.06 +
            chroma_score             * 0.06 +
            noise_inc_score          * 0.06
        )
        model_score = v1_raw / 1.00
        artifact_score = raw["compression"]
        temporal_score = raw.get("temporal", 0.0)

        final_score = (
            WEIGHT_MODEL       * model_score    +
            WEIGHT_ARTIFACT    * artifact_score +
            WEIGHT_TEMPORAL    * temporal_score +
            WEIGHT_EXPRESSION  * raw["expression"] +
            WEIGHT_METADATA    * meta_score     +
            WEIGHT_COMPRESSION * raw["compression"]
        )
    else:
        v1_raw = (
            raw["frequency"]         * 0.05 +
            raw["texture"]           * 0.08 +
            raw["edges"]             * 0.08 +
            raw["noise"]             * 0.06 +
            raw["gan_frequency"]     * 0.18 +
            raw["skin_tone"]         * 0.10 +
            raw["eye_artifacts"]     * 0.12 +
            raw["channel_decoupling"]* 0.10 +
            ela_score                * 0.07 +
            chroma_score             * 0.07 +
            noise_inc_score          * 0.07
        )
        v1_normalizer = 0.98

        total_raw = v1_raw
        total_normalizer = v1_normalizer
        model_score = total_raw / total_normalizer if total_normalizer > 0 else 0.0
        artifact_score = raw["compression"]

        final_score = (
            0.55 * model_score +
            0.25 * artifact_score +
            0.10 * meta_score +
            0.05 * ela_score +
            0.05 * noise_inc_score
        )

    final_score = float(np.clip(final_score, 0.0, 1.0))

    return final_score, signals, raw


# ─────────────────── CONTENT TYPE CLASSIFIER ─────────
# Runs before deepfake analysis — blocks AI-generated images and cartoons.
# Only real human photographs are allowed through.

class ContentTypeClassifier:
    """
    Detects whether an image is:
      - REAL_PHOTO      → real photograph of a person (pass)
      - AI_GENERATED    → generated by Stable Diffusion, Midjourney, DALL-E, etc.
      - CARTOON         → illustrated, anime, cartoon, or animated content

    Uses signal-based analysis only (no model download required):
      1. Color palette quantization  — cartoons/AI-art have unnaturally limited palettes
      2. Flat-region detection       — large unicolor blobs = cartoon/illustration
      3. Edge sharpness uniformity   — cartoon edges are artificially uniform
      4. Texture entropy             — AI-art/illustrations have low local texture entropy
      5. HSV saturation profile      — AI-art has hyper-vivid or uniform saturation
      6. High-frequency content      — natural photos have more high-freq detail
    """

    # Thresholds (tuned to balance FP vs FN on mixed photo/art sets)
    CARTOON_SCORE_THRESHOLD  = 0.65
    AI_GEN_SCORE_THRESHOLD   = 0.58

    def classify(self, image: np.ndarray):
        """Returns (content_type, score, signals) where content_type is
        'REAL_PHOTO', 'AI_GENERATED', or 'CARTOON'.
        """
        if image is None:
            return 'REAL_PHOTO', 0.0, []

        signals = []
        cartoon_score = 0.0
        ai_gen_score  = 0.0

        h, w = image.shape[:2]
        if h < 32 or w < 32:
            return 'REAL_PHOTO', 0.0, ['image_too_small_for_content_type_check']

        # ── 1. Color palette quantization ────────────────────
        # Cartoons and AI-art use a small number of dominant colors.
        # Real photos have hundreds of subtle color variations.
        small = cv2.resize(image, (128, 128))
        pixels = small.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        try:
            _, labels, _ = cv2.kmeans(pixels, 16, None, criteria,
                                       5, cv2.KMEANS_RANDOM_CENTERS)
            # How much of the image is covered by just 8 dominant clusters?
            label_counts = np.bincount(labels.flatten(), minlength=16)
            label_counts = np.sort(label_counts)[::-1]
            top8_coverage = label_counts[:8].sum() / (128 * 128)

            if top8_coverage > 0.92:   # ≥ 92% of pixels fall in top-8 colors
                cartoon_score += 0.35
                signals.append('limited_color_palette')
                if top8_coverage > 0.97:
                    cartoon_score += 0.15  # extremely quantized
        except Exception:
            pass  # k-means can fail on degenerate images

        # ── 2. Flat-region detection ──────────────────────────
        # Cartoons have large contiguous unicolor areas.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        # Large flat regions (>5% of total area)
        img_area = h * w
        large_region_count = sum(
            1 for i in range(1, num_labels)
            if stats[i, cv2.CC_STAT_AREA] > img_area * 0.05
        )
        if large_region_count >= 3:
            cartoon_score += 0.20
            signals.append('large_flat_regions')
        elif large_region_count >= 6:
            cartoon_score += 0.35

        # ── 3. Edge sharpness uniformity ─────────────────────
        # Cartoon edges are drawn with uniform sharpness.
        # Real photo edges have natural variation in sharpness.
        edges = cv2.Canny(gray, 50, 150).astype(np.float32)
        if edges.max() > 0:
            # Gradient magnitude at edge pixels
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            edge_mask = edges > 0
            if edge_mask.sum() > 100:
                edge_grads = grad_mag[edge_mask]
                edge_cv = np.std(edge_grads) / (np.mean(edge_grads) + 1e-8)
                if edge_cv < 0.30:   # very uniform edge intensity = cartoon
                    cartoon_score += 0.20
                    signals.append('uniform_edge_sharpness')
                elif edge_cv < 0.50:
                    cartoon_score += 0.08

        # ── 4. Local texture entropy ──────────────────────────
        # Real photos have high local texture entropy.
        # AI-art and illustrations are texturally smooth.
        from scipy.ndimage import uniform_filter  # always available with numpy
        try:
            gray_f = gray.astype(np.float32) / 255.0
            local_std = np.std(
                gray_f[
                    max(0, h//4):min(h, 3*h//4),
                    max(0, w//4):min(w, 3*w//4)
                ]
            )
            if local_std < 0.08:   # very smooth = likely AI/illustration
                ai_gen_score += 0.25
                cartoon_score += 0.20
                signals.append('low_texture_entropy')
            elif local_std < 0.12:
                ai_gen_score += 0.10
                cartoon_score += 0.08
        except ImportError:
            # scipy not available — use opencv equivalent
            local_region = gray[max(0, h//4):min(h, 3*h//4), max(0, w//4):min(w, 3*w//4)]
            local_std = np.std(local_region.astype(np.float32) / 255.0)
            if local_std < 0.08:
                ai_gen_score += 0.25
                cartoon_score += 0.20
                signals.append('low_texture_entropy')

        # ── 5. HSV saturation profile ─────────────────────────
        # AI-generated images often have hyper-saturated or unnaturally uniform
        # saturation distributions. Real photos have wide saturation spread.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        sat_std  = float(np.std(sat))
        sat_mean = float(np.mean(sat))

        # AI art: very high mean saturation (vivid) with low variance (uniform vividness)
        if sat_mean > 0.65 and sat_std < 0.18:
            ai_gen_score += 0.30
            signals.append('hyper_uniform_saturation')
        elif sat_mean > 0.55 and sat_std < 0.22:
            ai_gen_score += 0.15

        # ── 6. High-frequency content ratio ──────────────────
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        cy, cx = h // 2, w // 2
        low_r  = min(cy, cx) // 10
        high_r_start = int(min(cy, cx) * 0.40)
        y_grid, x_grid = np.ogrid[-cy:h-cy, -cx:w-cx]
        radius_grid = np.sqrt(y_grid**2 + x_grid**2)
        low_mask  = radius_grid <= low_r
        high_mask = radius_grid >= high_r_start
        low_energy  = magnitude[low_mask].mean()  + 1e-10
        high_energy = magnitude[high_mask].mean() + 1e-10
        hf_ratio = high_energy / low_energy

        if hf_ratio < 0.003:
            ai_gen_score += 0.30
            signals.append('abnormal_frequency_profile_low_hf')
        elif hf_ratio < 0.008:
            ai_gen_score += 0.15

        # ── 7. Upscaling artifact detection ─────────────────────
        # AI-generated images are often generated at low resolution then
        # upscaled. This creates characteristic interpolation artifacts
        # detectable in gradient histograms.
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        # Compute gradient magnitude histogram (32 bins)
        grad_hist, _ = np.histogram(grad_mag.ravel(), bins=32, range=(0, 255))
        grad_hist = grad_hist.astype(np.float32) / (grad_hist.sum() + 1e-10)
        # Entropy of gradient histogram: real photos have high entropy
        # (many different edge strengths); upscaled AI images have lower entropy
        grad_entropy = float(-np.sum(grad_hist * np.log2(grad_hist + 1e-10)))
        if grad_entropy < 3.0:
            ai_gen_score += 0.15
            signals.append('low_gradient_entropy_upscaled')
        elif grad_entropy < 3.8:
            ai_gen_score += 0.08

        # ── Decision ─────────────────────────────────────────
        # Cartoon score dominates if both are elevated
        if cartoon_score >= self.CARTOON_SCORE_THRESHOLD:
            return 'CARTOON', float(np.clip(cartoon_score, 0.0, 1.0)), signals
        if ai_gen_score >= self.AI_GEN_SCORE_THRESHOLD:
            return 'AI_GENERATED', float(np.clip(ai_gen_score, 0.0, 1.0)), signals
        return 'REAL_PHOTO', float(np.clip(max(ai_gen_score, cartoon_score), 0.0, 1.0)), signals


_CONTENT_TYPE_CLASSIFIER = ContentTypeClassifier()


# ─────────────────── MAIN ANALYSIS ───────────────────

def analyze(file_path):
    start_time = time.time()
    signals    = []

    video = _is_video(file_path)

    if video:
        frames = _sample_video_frames(file_path, n=MAX_FRAMES)
    else:
        img = _load_image(file_path)
        frames = [img] if img is not None else []

    if not frames:
        return _build_result(1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             ["media_decode_error", "fail_closed"], start_time, final_score=1.0)

    # ── Content Type Gate (AI-art & Cartoon blocker) ───
    # Only run on images — video reels are handled by reel_inference.py
    if not video and frames[0] is not None:
        content_type, ct_score, ct_signals = _CONTENT_TYPE_CLASSIFIER.classify(frames[0])
        if content_type == 'AI_GENERATED':
            _log(f"[ContentTypeClassifier] AI-generated image detected (score={ct_score:.3f})")
            result = _build_result(1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   ct_signals + ['ai_generated_image_blocked'],
                                   start_time, final_score=1.0)
            result['content_type']       = 'AI_GENERATED'
            result['content_type_score'] = round(ct_score, 4)
            result['verdict']            = 'REJECTED'
            return result
        elif content_type == 'CARTOON':
            _log(f"[ContentTypeClassifier] Cartoon/illustration detected (score={ct_score:.3f})")
            result = _build_result(1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   ct_signals + ['cartoon_illustration_blocked'],
                                   start_time, final_score=1.0)
            result['content_type']       = 'CARTOON'
            result['content_type_score'] = round(ct_score, 4)
            result['verdict']            = 'REJECTED'
            return result
        else:
            _log(f"[ContentTypeClassifier] Real photo confirmed (score={ct_score:.3f})")
            signals.append('content_type_real_photo')
            result_meta = {'content_type': 'REAL_PHOTO', 'content_type_score': round(ct_score, 4)}

    # ── Face Detection ────────────────────────────────────
    # has_faces: MTCNN-only (no Haar). Haar cascade produces massive false
    # positives on landscapes (40-50% of image), bypassing fail-closed.
    # MTCNN correctly rejects these same images.
    mtcnn_crops = _get_mtcnn_crops(frames)
    has_faces   = len(mtcnn_crops) >= (2 if video else 1)

    # For model inference: use all crops (MTCNN + Haar for better portrait recall)
    # Only bother computing if a face was confirmed by MTCNN.
    crops = _get_face_crops(frames) if has_faces else mtcnn_crops

    # ── Priority 1: ONNX model ensemble ────────────────
    onnx_score = None
    onnx_model_count = 0

    onnx_score, onnx_model_count, onnx_signal = _run_onnx_ensemble(frames)
    if onnx_score is not None:
        _log(f"[ONNX-Ensemble] {onnx_model_count} model(s) → P(fake)={onnx_score:.4f}")
        signals.append(onnx_signal)

    # ── Priority 2: Signal analysis (always run) ───────
    signal_score, signal_signals, raw = _run_signal_analysis(
        file_path, frames, crops, has_faces, video
    )
    signals.extend(signal_signals)

    # ── Score fusion ────────────────────────────────────
    if onnx_score is not None:
        signal_weight = 0.30
        model_weight = 0.70
        if video and raw.get("expression", 0.0) >= 0.55:
            signal_weight = 0.40
            model_weight = 0.60
        # If multiple models agree, trust them more
        if onnx_model_count >= 2:
            model_weight = min(0.80, model_weight + 0.05)
        final_score = model_weight * onnx_score + signal_weight * signal_score
        _log(f"[ONNX-Ensemble] fused score={final_score:.4f} "
             f"(model={onnx_score:.4f}, signals={signal_score:.4f})")
    else:
        # ── Priority 2: HuggingFace pre-trained model ──────
        hf_detector = _get_hf_detector()
        hf_score = None
        hf_temporal_score = None  # temporal variance across frames (video only)

        if hf_detector is not None and has_faces and crops:
            try:
                if video:
                    # VIDEO: sample more crops for temporal analysis
                    # Use up to 10 crops spread across the video
                    n_sample = min(10, len(crops))
                    step = max(1, len(crops) // n_sample)
                    sample_crops = crops[::step][:n_sample]
                    hf_scores = hf_detector.predict_batch(sample_crops)
                    if hf_scores:
                        hf_score = float(np.mean(hf_scores))
                        # Temporal variance: real videos have consistent P(fake) per frame
                        # Deepfakes flicker because the swap isn't temporally stable
                        # Threshold 0.22: stock footage with scene cuts can reach 0.15-0.20
                        # Genuine deepfakes flip dramatically (std > 0.22) frame-to-frame
                        if len(hf_scores) >= 4:
                            hf_temporal_score = float(np.std(hf_scores))
                            # Require both high temporal variance AND elevated mean score
                            # to avoid false positives from scene cuts in stock footage
                            if hf_temporal_score > 0.22 and hf_score >= 0.45:
                                signals.append("hf_temporal_instability")
                                _log(f"[HuggingFace] temporal std={hf_temporal_score:.4f} "
                                     f"mean={hf_score:.4f} (>0.22 + mean>=0.45 = deepfake instability)")
                            elif hf_temporal_score > 0.22:
                                _log(f"[HuggingFace] temporal std={hf_temporal_score:.4f} HIGH "
                                     f"but mean={hf_score:.4f} too low — scene cuts, not deepfake")
                        # Max-score: a single frame with very high P(fake) is suspicious
                        # Only flag if the AVERAGE is also elevated (otherwise it's one bad frame)
                        hf_max_score = float(np.max(hf_scores))
                        if hf_max_score > 0.85 and hf_score >= 0.55:
                            signals.append("hf_high_confidence_frame")
                            _log(f"[HuggingFace] max frame score={hf_max_score:.4f} avg={hf_score:.4f} (>0.85 + avg>=0.55)")
                        elif hf_max_score > 0.85:
                            _log(f"[HuggingFace] max frame score={hf_max_score:.4f} but avg={hf_score:.4f} too low — not flagged")
                        signals.append("huggingface_model_used")
                        _log(f"[HuggingFace] video P(fake)={hf_score:.4f} "
                             f"max={hf_max_score:.4f} "
                             f"(avg over {len(sample_crops)} crops)")
                else:
                    # IMAGE: run dima806 on full image and on face crops
                    hf_full_score = hf_detector.predict(frames[0]) if (frames and frames[0] is not None) else 0.0
                    sample_crops = crops[:5]
                    hf_scores = hf_detector.predict_batch(sample_crops)
                    hf_crop_score = float(np.mean(hf_scores)) if hf_scores else 0.0
                    
                    hf_score = hf_full_score
                    signals.append("huggingface_model_used")
                    _log(f"[HuggingFace] Image P(fake): full={hf_full_score:.4f}, crop={hf_crop_score:.4f} (avg over {len(sample_crops)} crops)")
                    
                    # Smart Full/Crop Fusion:
                    # Real portraits have full_prob < 0.01, StyleGAN fakes have full_prob >= 0.04.
                    # Crop score >= 0.75 reliably indicates fake face structure.
                    if hf_full_score >= 0.04 and hf_crop_score >= 0.75:
                        hf_score = max(hf_full_score, 0.75)
                        signals.append("gan_ensemble_model_used")
                        _log(f"[HuggingFace] Smart Fusion: full={hf_full_score:.4f}, crop={hf_crop_score:.4f} -> REJECTED")

            except Exception as hf_err:
                _log(f"[HuggingFace] inference error: {hf_err}")
                hf_score = None

        if hf_score is not None:
            if video:
                # ── VIDEO: stricter fusion — signals are very reliable for deepfake reels
                # The HF model sees individual frames without temporal context.
                # For videos, trust signals more (40%) and add temporal variance bonus.
                # Deepfake signals on videos (skin_tone_instability, temporal_face_distortion)
                # are MUCH more reliable than on still images.
                video_signal_weight = 0.45
                video_model_weight = 0.55

                # If temporal instability confirmed (not just scene cuts), signals get more weight
                if "hf_temporal_instability" in signals:
                    video_signal_weight = 0.55
                    video_model_weight = 0.45

                final_score = video_model_weight * hf_score + video_signal_weight * signal_score

                # Temporal instability bonus: deepfakes flicker between frames
                # Only applies when hf_temporal_instability is confirmed (std>0.22 AND mean>=0.45)
                if "hf_temporal_instability" in signals and hf_temporal_score is not None:
                    temporal_bonus = min(0.15, (hf_temporal_score - 0.22) * 2.0)
                    final_score = float(np.clip(final_score + temporal_bonus, 0.0, 1.0))
                    _log(f"[HuggingFace] temporal bonus={temporal_bonus:.4f} (std={hf_temporal_score:.4f})")

                # High-confidence single frame: bumps score up toward rejection
                if "hf_high_confidence_frame" in signals:
                    hf_max_score = float(max(
                        [hf_detector.predict(c) for c in crops[-3:] if c is not None]
                        or [hf_score]
                    ))
                    frame_bump = min(0.10, (hf_max_score - 0.85) * 0.8)
                    final_score = float(np.clip(final_score + frame_bump, 0.0, 1.0))

                _log(f"[HuggingFace] video fused score={final_score:.4f} "
                     f"(hf={hf_score:.4f}, signals={signal_score:.4f})")
            else:
                # ── IMAGE: HF confidence-corroboration adjustment ──────────────────
                # dima806 is biased toward labeling professional stock photos as fake.
                # If HF says fake (>= 0.65) but NO genuinely deepfake-specific signals
                # corroborate it, apply a heavy penalty and ignore signal_score.
                #
                # DEEPFAKE_SPECIFIC: signals that are reliable indicators of AI fakery.
                # These bypass the penalty gate and count as corroboration.
                DEEPFAKE_SPECIFIC = {
                    "skin_tone_instability",
                    "temporal_face_distortion",
                    "facial_expression_inconsistency",
                    "gan_ensemble_model_used",
                    "gan_spectral_fingerprint",
                    "eye_region_gan_artifact",
                    "enhanced_blending_seam",
                }
                strong_corr = len(set(signals) & DEEPFAKE_SPECIFIC)

                # Count v2-specific signals for adaptive weighting
                V2_SIGNALS = {"enhanced_blending_seam"}
                v2_fired = len(set(signals) & V2_SIGNALS)

                if "gan_ensemble_model_used" in signals:
                    # GAN-specific model caught this — trust it at 100%, ignore low signal_score.
                    # StyleGAN faces have very clean signals (no JPEG noise) so signal_score is low.
                    # BUT: if dima806 is firing on pro photography (false positive), the signal_score
                    # will also be low and no genuine deepfake signals will corroborate.
                    # Check: do we have corroborating deepfake-specific signals?
                    NON_CIRCULAR = {s for s in DEEPFAKE_SPECIFIC if s != "gan_ensemble_model_used"}
                    if len(set(signals) & NON_CIRCULAR) >= 1:
                        # Corroborated by at least one non-circular deepfake signal
                        final_score = hf_score
                        _log(f"[GAN-Detector] GAN-corroborated score={final_score:.4f} "
                             f"(gan_ensemble={hf_score:.4f}, corr={set(signals) & NON_CIRCULAR})")
                    else:
                        # No corroboration — may be dima806 false positive on real photography
                        # Use adaptive fusion to let signal_score weigh in
                        hf_adjusted = hf_score
                        fused = 0.70 * hf_adjusted + 0.30 * signal_score
                        # Apply mild penalty: dima806 has known FP rate on studio/portrait photos
                        penalty = 0.15  # reduces final score by 0.15
                        final_score = float(np.clip(fused - penalty, 0.0, 1.0))
                        _log(f"[GAN-Detector] GAN-uncorroborated score={final_score:.4f} "
                             f"(hf={hf_adjusted:.4f}, signals={signal_score:.4f}, "
                             f"penalty={penalty:.2f})")
                elif hf_score >= 0.65 and strong_corr == 0:
                    # Heavy penalty: dima806 is firing on professional photography.
                    # -0.52 penalty: hf_score=0.91 -> 0.39 (APPROVED).
                    hf_adjusted = float(np.clip(hf_score - 0.52, 0.0, 1.0))
                    final_score = hf_adjusted
                    signals.append("hf_confidence_penalty_applied")
                    _log(f"[HuggingFace] confidence penalty: {hf_score:.4f} -> {hf_adjusted:.4f} "
                         f"(signal_score={signal_score:.4f} excluded)")
                else:
                    hf_adjusted = hf_score
                    # Adaptive fusion: v2 signals provide strong evidence even when
                    # the HF model (trained on face-swap only) returns low scores.
                    V2_SIGNAL_STRONG = v2_fired >= 2 or (v2_fired >= 1 and signal_score >= 0.25)

                    if V2_SIGNAL_STRONG:
                        # Multiple v2 signals agree → strong signal-based evidence.
                        # Even with low HF score, the signal ensemble is reliable.
                        signal_weight = 0.60
                        model_weight = 0.40
                        signals.append("v2_signal_ensemble_boost")
                    elif v2_fired >= 1:
                        signal_weight = 0.40
                        model_weight = 0.60
                    else:
                        signal_weight = 0.30
                        model_weight = 0.70

                    fused = model_weight * hf_adjusted + signal_weight * signal_score

                    # Floor: if 2+ v2 signals fire, don't let a near-zero HF score
                    # suppress clear signal evidence of deepfake artifacts.
                    # The ensemble of independent v2 detectors provides reliable detection.
                    if v2_fired >= 3:
                        fused = max(fused, signal_score * 0.90)
                    elif v2_fired >= 2:
                        fused = max(fused, signal_score * 0.85)
                    elif v2_fired >= 1 and signal_score >= 0.20:
                        fused = max(fused, signal_score * 0.70)

                    final_score = float(np.clip(fused, 0.0, 1.0))
                    _log(f"[HuggingFace] fused score={final_score:.4f} "
                         f"(hf={hf_adjusted:.4f}, signals={signal_score:.4f}, "
                         f"w_sig={signal_weight:.2f})")
        else:
            # Pure signal fallback (last resort -- no ML model available)
            final_score = signal_score
            signals.append("signal_analysis_fallback")
            _log(f"[SignalAnalysis] fallback score={final_score:.4f}")

    # ── Deepfake Signal Boosting ────────────────────────
    # When HF/ONNX model is active: ONLY boost on deepfake-specific signals.
    # JPEG-noise signals (face_blending_seam, color_channel_decoupled, etc.)
    # trigger constantly on real Pexels/Unsplash photos and must not be used
    # as boosters when a model score is available.
    # When pure signal fallback: use all signals.
    _hf_active   = "huggingface_model_used"      in signals
    _onnx_active = "onnx_model_ensemble" in signals
    _model_active = _hf_active or _onnx_active

    boost = 0.0
    if _model_active:
        if video:
            if "temporal_face_distortion"        in signals: boost += 0.14
            if "facial_expression_inconsistency" in signals: boost += 0.14
            if "hf_temporal_instability"         in signals: boost += 0.12
            boost = min(boost, 0.25)
        else:
            if "skin_tone_instability"            in signals: boost += 0.10
            if "gan_spectral_fingerprint"         in signals: boost += 0.12
            if "eye_region_gan_artifact"          in signals: boost += 0.10
            if "face_blending_seam"               in signals: boost += 0.08
            if "enhanced_blending_seam"           in signals: boost += 0.18
            if "ela_recompression_artifact"       in signals: boost += 0.12
            if "chromatic_aberration_anomaly"     in signals: boost += 0.08
            if "noise_pattern_inconsistency"      in signals: boost += 0.10
            if "color_channel_decoupled"          in signals: boost += 0.08
            if "dct_coefficient_anomaly"          in signals: boost += 0.10
            if "wavelet_texture_artifact"         in signals: boost += 0.08
            if "reencoding_block_artifacts"       in signals: boost += 0.08
            if "gan_ensemble_model_used"          in signals:
                NON_CIRCULAR = {"gan_spectral_fingerprint", "eye_region_gan_artifact",
                                "enhanced_blending_seam", "ela_recompression_artifact",
                                "noise_pattern_inconsistency"}
                if len(set(signals) & NON_CIRCULAR) >= 1:
                    boost += 0.15
                else:
                    boost += 0.08
            boost = min(boost, 0.50)
    elif has_faces:
        if "gan_spectral_fingerprint"    in signals: boost += 0.18
        if "face_blending_seam"          in signals: boost += 0.15
        if "enhanced_blending_seam"      in signals: boost += 0.20
        if "eye_region_gan_artifact"     in signals: boost += 0.15
        if "ela_recompression_artifact"  in signals: boost += 0.15
        if "noise_pattern_inconsistency" in signals: boost += 0.12
        if "chromatic_aberration_anomaly" in signals: boost += 0.10
        if "unnatural_face_texture"      in signals: boost += 0.10
        if "color_channel_decoupled"     in signals: boost += 0.10
        if "dct_coefficient_anomaly"     in signals: boost += 0.12
        if "wavelet_texture_artifact"    in signals: boost += 0.10
        if "oversmoothed_skin_detected"  in signals: boost += 0.10
        if "oversmoothed_blur_artifact"  in signals: boost += 0.10
        if "abnormal_noise_pattern"      in signals: boost += 0.08
        if "reencoding_block_artifacts"  in signals: boost += 0.08
        boost = min(boost, 0.55)
    else:
        if "gan_spectral_fingerprint"   in signals: boost += 0.12
        if "color_channel_decoupled"    in signals: boost += 0.08
        if "ela_recompression_artifact" in signals: boost += 0.10
        if "wavelet_texture_artifact"   in signals: boost += 0.08
        boost = min(boost, 0.30)

    final_score = float(np.clip(final_score + boost, 0.0, 1.0))

    result = _build_result(
        raw.get("frequency", 0.0),
        raw.get("compression", 0.0),
        raw.get("temporal", 0.0),
        raw.get("expression", 0.0),
        0.0,
        raw.get("compression", 0.0),
        signals,
        start_time,
        final_score,
        onnx_score,
        is_video=video,
    )
    # Attach content-type metadata if classifier ran
    if not video and 'result_meta' in dir():
        result.update(result_meta)
    return result


def _build_result(model_score, artifact_score, temporal_score,
                  expression_score, metadata_score, compression_score, signals,
                  start_time, final_score=None, onnx_score=None, is_video=False):
    if final_score is None:
        final_score = (
            WEIGHT_MODEL       * model_score    +
            WEIGHT_ARTIFACT    * artifact_score +
            WEIGHT_TEMPORAL    * temporal_score +
            WEIGHT_EXPRESSION  * expression_score +
            WEIGHT_METADATA    * metadata_score +
            WEIGHT_COMPRESSION * compression_score
        )
    final_score = float(np.clip(final_score, 0.0, 1.0))

    if is_video:
        threshold_reject  = 0.60
        threshold_approve = THRESHOLD_APPROVE
    else:
        threshold_reject  = THRESHOLD_REJECT
        threshold_approve = THRESHOLD_APPROVE

    if final_score >= threshold_reject:
        verdict = "REJECTED"
        if "deepfake_detected" not in signals:
            signals.append("synthetic_generation_signal")
    elif final_score >= threshold_approve:
        verdict = "UNDER_REVIEW"
        if "borderline_needs_review" not in signals:
            signals.append("borderline_needs_review")
    else:
        verdict = "APPROVED"
        signals = [s for s in signals if s not in
                   ("synthetic_generation_signal", "deepfake_detected", "borderline_needs_review")]

    result = {
        "model":             "lightfakedetect-v2",
        "model_score":       round(float(model_score), 4),
        "artifact_score":    round(float(artifact_score), 4),
        "temporal_score":    round(float(temporal_score), 4),
        "expression_score":  round(float(expression_score), 4),
        "metadata_score":    round(float(metadata_score), 4),
        "compression_score": round(float(compression_score), 4),
        "final_score":       round(final_score, 4),
        "verdict":           verdict,
        "signals":           list(set(signals)),
    }
    if onnx_score is not None:
        result["onnx_model_score"] = round(float(onnx_score), 4)
    if is_video:
        result["video_threshold_reject"] = 0.60

    return result


# ─────────────────── ENTRY POINT ─────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({
            "model": "lightfakedetect-v2",
            "model_score": 0.1, "artifact_score": 0.0,
            "temporal_score": 0.0, "expression_score": 0.0, "metadata_score": 0.0,
            "compression_score": 0.0, "final_score": 0.1,
            "verdict": "REJECTED", "signals": ["no_file_provided"],
        }))
        sys.exit(0)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(json.dumps({
            "model": "lightfakedetect-v2",
            "model_score": 0.1, "artifact_score": 0.0,
            "temporal_score": 0.0, "expression_score": 0.0, "metadata_score": 0.0,
            "compression_score": 0.0, "final_score": 0.1,
            "verdict": "REJECTED",
            "signals": [f"file_not_found: {file_path}", "fail_closed"],
        }))
        sys.exit(0)

    try:
        result = analyze(file_path)
        print(json.dumps(result))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({
            "model": "lightfakedetect-v2",
            "model_score": 0.1, "artifact_score": 0.0,
            "temporal_score": 0.0, "expression_score": 0.0, "metadata_score": 0.0,
            "compression_score": 0.0, "final_score": 0.1,
            "verdict": "REJECTED",
            "signals": [f"engine_error: {str(e)}", "fail_closed"],
        }))
        sys.exit(0)
