"""
TrueFrame Reels — Simple Deepfake Video Detector
==================================================
Detects deepfake videos using signal analysis with OpenCV + NumPy only.
No ONNX model, no PyTorch, no model files required.

Signals analysed:
  1. Temporal flicker  — unnatural frame-to-frame brightness jumps
  2. Block artifacts   — JPEG/compression block grid patterns common in re-encoded fakes
  3. Face-crop variance— real faces have natural texture variance; fakes tend to be smooth
  4. Blending edges    — gradient discontinuities around face boundary (GAN paste seam)
  5. Color constancy   — abrupt color temperature shifts between frames
  6. Noise floor       — GAN-generated faces have unusually low or high noise texture

Usage:
    python reel_inference.py <video_path>

Output: JSON with deepfake probability, verdict, and signals.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger("trueframe.inference")

# ─────────────────── THRESHOLDS ──────────────────────
THRESHOLD_APPROVE = 0.80   # < 0.80 → APPROVED (binary decision)
THRESHOLD_REVIEW  = 0.80   # unused sentinel — kept for schema compatibility
THRESHOLD_REJECT  = 0.80   # >= 0.80 → REJECTED

MAX_FRAMES = 20
FRAME_SIZE = (224, 224)


# ─────────────────── HELPERS ─────────────────────────

def _open_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0, 0
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, total


def _sample_frames(video_path, n=MAX_FRAMES):
    """Return up to n evenly-spaced BGR frames from the video."""
    cap, fps, total = _open_video(video_path)
    if cap is None:
        return []
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


def _detect_face_haar(frame):
    """Fallback face detection with Haar cascade — always available via OpenCV."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    return frame[y:y + h, x:x + w]


def _detect_face_mediapipe(frame, face_det):
    """MediaPipe-based face detection (optional)."""
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_det.process(rgb)
    if not results.detections:
        return None
    best = max(results.detections, key=lambda d: d.score[0])
    bbox = best.location_data.relative_bounding_box
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox.xmin * w))
    y1 = max(0, int(bbox.ymin * h))
    x2 = min(w, x1 + int(bbox.width * w))
    y2 = min(h, y1 + int(bbox.height * h))
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return frame[y1:y2, x1:x2]


def _get_face_detector():
    """Return (detect_fn) that accepts a BGR frame and returns a face crop or None."""
    try:
        import mediapipe as mp
        try:
            face_det = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
        except Exception:
            face_det = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

        def detect(frame):
            return _detect_face_mediapipe(frame, face_det)
        return detect
    except Exception:
        return _detect_face_haar


def _extract_face_crops(frames, detect_fn):
    """Extract & resize face crops from frames list."""
    crops = []
    for frame in frames:
        face = detect_fn(frame)
        if face is not None and face.size > 0:
            face_resized = cv2.resize(face, FRAME_SIZE)
            crops.append(face_resized)
    return crops


# ─────────────────── SIGNAL ANALYSIS ─────────────────

def signal_temporal_flicker(frames):
    """
    Measure unnatural brightness flicker between consecutive frames.
    Real videos have smooth brightness changes; deepfakes often stutter.
    Score: 0 (normal) → 1 (extreme flicker).
    """
    if len(frames) < 4:
        return 0.0, False
    brightnesses = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() for f in frames]
    diffs = np.abs(np.diff(brightnesses))
    mean_diff  = float(np.mean(diffs))
    max_diff   = float(np.max(diffs))
    # Typical real video: mean_diff < 3, max_diff < 15
    score = min(1.0, (mean_diff / 6.0) * 0.4 + (max_diff / 30.0) * 0.6)
    triggered = mean_diff > 5.0 or max_diff > 20.0
    return score, triggered


def signal_block_artifacts(frames):
    """
    Detect 8×8 DCT compression grid artifacts common in re-encoded deepfakes.
    Score: 0 (clean) → 1 (heavy artifacts).
    NOTE: All real camera videos have some DCT blocking when encoded as mp4/h264.
    Only flag extreme re-encoding artifacts, not normal camera compression.
    """
    if not frames:
        return 0.0, False
    scores = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Detect horizontal 8-pixel periodicity via DFT
        h, w = gray.shape
        cols = gray[:, :w - (w % 8)]
        row_means = cols.reshape(h, -1, 8).mean(axis=2)  # (H, N_blocks)
        block_variance = float(np.var(row_means))
        pixel_variance  = float(np.var(gray))
        ratio = block_variance / (pixel_variance + 1e-6)
        scores.append(ratio)
    mean_ratio = float(np.mean(scores))
    # Real camera videos: ratio typically 0.03–0.35
    # Heavily re-encoded fake videos: ratio > 0.50
    score = min(1.0, max(0.0, (mean_ratio - 0.35) / 0.40))
    triggered = mean_ratio > 0.45
    return score, triggered


def signal_face_texture_variance(crops):
    """
    GAN-generated faces are often unnaturally smooth (low Laplacian variance)
    OR overly sharp (very high). Score both extremes.
    Real faces: 100 < mean_var < 6000 (wide range due to focus, lighting).
    Only flag truly extreme cases.
    """
    if not crops:
        return 0.0, False
    variances = []
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap  = cv2.Laplacian(gray, cv2.CV_64F)
        variances.append(float(np.var(lap)))
    mean_var = float(np.mean(variances))
    std_var  = float(np.std(variances))
    # Real faces typically: 100 < mean_var < 6000 and std_var relatively stable
    too_smooth    = mean_var < 30.0
    too_sharp     = mean_var > 15000.0
    unstable_var  = std_var / (mean_var + 1.0) > 2.5
    triggered = too_smooth or too_sharp or unstable_var
    if too_smooth:
        score = min(1.0, 30.0 / (mean_var + 1.0))
    elif too_sharp:
        score = min(1.0, (mean_var - 15000.0) / 10000.0)
    elif unstable_var:
        score = min(1.0, (std_var / (mean_var + 1.0) - 2.5) / 2.0)
    else:
        score = 0.0
    return score, triggered


def signal_blending_edges(crops):
    """
    GAN face-swap artifacts: edge sharpness discontinuity at face boundary.
    Measure Sobel gradient near the border vs interior of the face crop.
    """
    if not crops:
        return 0.0, False
    edge_ratios = []
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sobel = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        )
        h, w  = sobel.shape
        border_width = max(8, int(min(h, w) * 0.1))
        # Border strip
        border = np.concatenate([
            sobel[:border_width, :].ravel(),
            sobel[-border_width:, :].ravel(),
            sobel[:, :border_width].ravel(),
            sobel[:, -border_width:].ravel(),
        ])
        interior = sobel[border_width:-border_width, border_width:-border_width].ravel()
        if interior.size == 0:
            continue
        ratio = float(np.mean(border)) / (float(np.mean(interior)) + 1e-6)
        edge_ratios.append(ratio)
    if not edge_ratios:
        return 0.0, False
    mean_ratio = float(np.mean(edge_ratios))
    # Normal faces: border ≈ interior → ratio near 1.0
    # GAN paste seam: ratio > 1.6 or < 0.5
    deviation = abs(mean_ratio - 1.0)
    triggered = deviation > 0.5
    score = min(1.0, deviation / 1.2)
    return score, triggered


def signal_color_consistency(frames):
    """
    Detect abrupt color temperature shifts between frames.
    Real videos maintain consistent white balance; deepfakes flicker in hue.
    """
    if len(frames) < 4:
        return 0.0, False
    hue_means = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue_means.append(float(hsv[:, :, 0].mean()))
    diffs = np.abs(np.diff(hue_means))
    mean_diff = float(np.mean(diffs))
    max_diff  = float(np.max(diffs))
    # Typical real: mean_diff < 4, max_diff < 20
    score = min(1.0, (mean_diff / 8.0) * 0.5 + (max_diff / 40.0) * 0.5)
    triggered = mean_diff > 5.0 or max_diff > 25.0
    return score, triggered


def signal_noise_floor(crops):
    """
    GAN images have a characteristic noise pattern. Estimate local noise
    using high-frequency residual after Gaussian blur.
    Real faces after camera capture: mean_noise typically 0.8–3.5.
    GAN faces: often < 0.4 (unnaturally clean after post-processing).
    """
    if not crops:
        return 0.0, False
    noise_levels = []
    for crop in crops:
        gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        residual = np.abs(gray - blurred)
        noise_levels.append(float(np.mean(residual)))
    mean_noise = float(np.mean(noise_levels))
    std_noise  = float(np.std(noise_levels))
    # Only flag truly extreme: below 0.4 (no real camera produces this) or very inconsistent
    too_clean      = mean_noise < 0.4
    inconsistent   = std_noise / (mean_noise + 1e-6) > 1.5
    triggered = too_clean or inconsistent
    if too_clean:
        score = min(1.0, 0.4 / (mean_noise + 0.05))
    elif inconsistent:
        score = min(1.0, (std_noise / (mean_noise + 1e-6) - 1.5) / 2.0)
    else:
        score = 0.0
    return score, triggered


# ─────────────────── MAIN ENGINE ─────────────────────

def analyze_video(video_path):
    """
    Analyze a video file for deepfake content.
    Returns a JSON-compatible dict with scores, verdict, and signals.
    """
    start_time = time.time()
    signals    = []

    # ── 1. Sample frames ──────────────────────────────
    frames = _sample_frames(video_path, n=MAX_FRAMES)
    if len(frames) < 3:
        return _build_result(0.5, ["insufficient_frames"], start_time)

    # ── 2. Extract face crops ─────────────────────────
    detect_fn = _get_face_detector()
    crops     = _extract_face_crops(frames, detect_fn)

    has_faces = len(crops) >= 3

    # ── 3. Run signal analysis ────────────────────────
    raw_scores = {}

    # Temporal signals (work on raw frames)
    flicker_score, flicker_triggered = signal_temporal_flicker(frames)
    raw_scores["temporal_flicker"] = flicker_score
    if flicker_triggered:
        signals.append("temporal_flicker_detected")

    block_score, block_triggered = signal_block_artifacts(frames)
    raw_scores["block_artifacts"] = block_score
    if block_triggered:
        signals.append("compression_block_artifacts")

    color_score, color_triggered = signal_color_consistency(frames)
    raw_scores["color_shift"] = color_score
    if color_triggered:
        signals.append("color_temperature_inconsistency")

    # Face-based signals (only when faces found)
    if has_faces:
        texture_score, texture_triggered = signal_face_texture_variance(crops)
        raw_scores["face_texture"] = texture_score
        if texture_triggered:
            signals.append("unnatural_face_texture")

        edge_score, edge_triggered = signal_blending_edges(crops)
        raw_scores["blending_edges"] = edge_score
        if edge_triggered:
            signals.append("face_blending_seam")

        noise_score, noise_triggered = signal_noise_floor(crops)
        raw_scores["noise_floor"] = noise_score
        if noise_triggered:
            signals.append("abnormal_noise_pattern")
    else:
        # No faces detected — do NOT assign suspicious scores.
        # Many real videos have no faces (scenery, sports, etc.).
        signals.append("no_face_detected")
        raw_scores["face_texture"]   = 0.0
        raw_scores["blending_edges"] = 0.0
        raw_scores["noise_floor"]    = 0.0

    # ── 4. Weighted fusion ────────────────────────────
    # Weights tuned so face signals dominate when available
    if has_faces:
        weights = {
            "temporal_flicker": 0.15,
            "block_artifacts":  0.15,
            "color_shift":      0.10,
            "face_texture":     0.25,
            "blending_edges":   0.20,
            "noise_floor":      0.15,
        }
    else:
        weights = {
            "temporal_flicker": 0.30,
            "block_artifacts":  0.35,
            "color_shift":      0.20,
            "face_texture":     0.05,
            "blending_edges":   0.05,
            "noise_floor":      0.05,
        }

    final_score = float(sum(
        raw_scores.get(k, 0.0) * w for k, w in weights.items()
    ))
    final_score = round(min(1.0, max(0.0, final_score)), 4)

    return _build_result(final_score, signals, start_time, raw_scores)


def _build_result(prob, signals, start_time, raw_scores=None):
    elapsed_ms = (time.time() - start_time) * 1000
    prob       = round(float(np.clip(prob, 0, 1)), 4)
    authenticity = round(1.0 - prob, 4)

    if prob >= THRESHOLD_REJECT:
        verdict    = "REJECTED"
        confidence = "HIGH"
        if "deepfake_detected" not in signals:
            signals.append("deepfake_detected")
    else:
        verdict    = "APPROVED"
        confidence = "HIGH"
        # Remove any leftover rejection signals from a previous run
        signals = [s for s in signals if s not in ("deepfake_detected",)]

    result = {
        "model":              "trueframe-video-signal-analyzer",
        "model_score":        prob,
        "artifact_score":     round(float((raw_scores or {}).get("block_artifacts", 0.0)), 4),
        "temporal_score":     round(float((raw_scores or {}).get("temporal_flicker", 0.0)), 4),
        "metadata_score":     0.0,
        "compression_score":  round(float((raw_scores or {}).get("block_artifacts", 0.0)), 4),
        "final_score":        prob,
        "deepfake_probability": prob,
        "authenticity_score": authenticity,
        "verdict":            verdict,
        "confidence":         confidence,
        "inference_time_ms":  round(elapsed_ms, 1),
        "signals":            list(set(signals)),
    }
    return result


# ─────────────────── CLI ENTRY POINT ─────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    if len(sys.argv) < 2:
        print(json.dumps({
            "model":              "trueframe-video-signal-analyzer",
            "model_score":        0.5,
            "artifact_score":     0.0,
            "temporal_score":     0.0,
            "metadata_score":     0.0,
            "compression_score":  0.0,
            "final_score":        0.5,
            "deepfake_probability": 0.5,
            "authenticity_score": 0.5,
            "verdict":            "REJECTED",
            "signals":            ["no_file_provided"],
        }))
        sys.exit(1)

    try:
        result = analyze_video(sys.argv[1])
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({
            "model":              "trueframe-video-signal-analyzer",
            "model_score":        0.5,
            "artifact_score":     0.0,
            "temporal_score":     0.0,
            "metadata_score":     0.0,
            "compression_score":  0.0,
            "final_score":        0.5,
            "deepfake_probability": 0.5,
            "authenticity_score": 0.5,
            "verdict":            "REJECTED",
            "signals":            [f"engine_error: {str(e)}", "fail_closed"],
        }))
        sys.exit(1)
