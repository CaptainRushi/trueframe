"""
TrueFrame AI Service — Simple Deepfake Detector
=================================================
Detects deepfake images and videos using signal analysis only.
No ONNX model, no HuggingFace, no model files required.
Works on any machine with OpenCV + NumPy installed.

Signals analysed:
  1. Frequency artifacts  — GAN grid noise in DCT/FFT domain
  2. Block artifacts      — 8×8 JPEG compression grid patterns
  3. Face texture variance — unnatural smoothness or sharpness
  4. Blending edge seam   — gradient discontinuities at face boundary
  5. Noise floor          — abnormally low/inconsistent image noise
  6. Color consistency    — abrupt hue/white-balance shifts (video)
  7. Temporal flicker     — frame-to-frame brightness jumps (video)
  8. Metadata score       — lightweight header check

Usage:
    python main.py <file_path>

Output: JSON with scores and verdict (same schema as before).
"""

import sys
import os
import json
import time
import numpy as np
import cv2

# ─────────────────── CONFIG ──────────────────────────
WEIGHT_MODEL        = 0.40
WEIGHT_ARTIFACT     = 0.20
WEIGHT_TEMPORAL     = 0.15
WEIGHT_METADATA     = 0.10
WEIGHT_COMPRESSION  = 0.15

THRESHOLD_APPROVE   = 0.60   # < 0.60 → APPROVED
THRESHOLD_REJECT    = 0.80   # >= 0.80 → REJECTED

MAX_FRAMES          = 20
FRAME_SIZE          = (224, 224)


# ─────────────────── HELPERS ─────────────────────────

def _log(msg):
    print(msg, file=sys.stderr)


def _is_video(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.m4v', '.3gp'}


def _load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
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


def _detect_faces_haar(frame):
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    crops = []
    for (x, y, w, h) in faces:
        crops.append(frame[y:y + h, x:x + w])
    return crops


def _get_face_crops(frames):
    """Try MediaPipe first, fall back to Haar."""
    face_det = None
    try:
        import mediapipe as mp
        face_det = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
    except Exception:
        pass

    crops = []
    for frame in frames:
        if face_det is not None:
            try:
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_det.process(rgb)
                if results.detections:
                    best = max(results.detections, key=lambda d: d.score[0])
                    bbox = best.location_data.relative_bounding_box
                    h, w = frame.shape[:2]
                    x1 = max(0, int(bbox.xmin * w))
                    y1 = max(0, int(bbox.ymin * h))
                    x2 = min(w, x1 + int(bbox.width * w))
                    y2 = min(h, y1 + int(bbox.height * h))
                    if x2 - x1 >= 32 and y2 - y1 >= 32:
                        crops.append(cv2.resize(frame[y1:y2, x1:x2], FRAME_SIZE))
                    continue
            except Exception:
                pass
        # Haar fallback
        haar_crops = _detect_faces_haar(frame)
        for c in haar_crops:
            if c.shape[0] >= 32 and c.shape[1] >= 32:
                crops.append(cv2.resize(c, FRAME_SIZE))
    return crops


# ─────────────────── SIGNAL DETECTORS ────────────────

def _signal_frequency_artifacts(frames):
    """
    GAN generators leave characteristic periodic noise in frequency domain.
    Measure power in mid-high frequency bands via 2D FFT.
    Score: 0 (natural) → 1 (suspicious).
    """
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
        # Annular band: 10% – 40% of Nyquist
        r_inner = int(min(h, w) * 0.10)
        r_outer = int(min(h, w) * 0.40)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        band_mask = (dist >= r_inner) & (dist <= r_outer)
        band_power = magnitude[band_mask].mean()
        total_power = magnitude.mean()
        ratio = band_power / (total_power + 1e-6)
        scores.append(ratio)
    mean_ratio = float(np.mean(scores))
    # GAN images typically: ratio > 0.95 (unnaturally uniform mid-band)
    score = min(1.0, max(0.0, (mean_ratio - 0.80) / 0.25))
    triggered = mean_ratio > 0.90
    return score, triggered


def _signal_block_artifacts(frames):
    """Detect 8×8 DCT block grid from re-encoding."""
    if not frames:
        return 0.0, False
    scores = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        # Measure 8-pixel column periodicity
        cols = gray[:, :w - (w % 8)]
        if cols.shape[1] < 8:
            scores.append(0.0)
            continue
        row_means = cols.reshape(h, -1, 8).mean(axis=2)
        block_var = float(np.var(row_means))
        pixel_var = float(np.var(gray))
        scores.append(block_var / (pixel_var + 1e-6))
    mean_ratio = float(np.mean(scores))
    score = min(1.0, mean_ratio / 0.25)
    triggered = mean_ratio > 0.15
    return score, triggered


def _signal_face_texture(crops):
    """Unnatural face smoothness or sharpness via Laplacian variance."""
    if not crops:
        return 0.0, False
    variances = []
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap  = cv2.Laplacian(gray, cv2.CV_64F)
        variances.append(float(np.var(lap)))
    mean_var = float(np.mean(variances))
    std_var  = float(np.std(variances))
    too_smooth   = mean_var < 80.0
    too_sharp    = mean_var > 8000.0
    unstable     = std_var / (mean_var + 1.0) > 1.5
    triggered = too_smooth or too_sharp or unstable
    if too_smooth:
        score = min(1.0, 80.0 / (mean_var + 1.0))
    elif too_sharp:
        score = min(1.0, mean_var / 12000.0)
    elif unstable:
        score = min(1.0, std_var / (mean_var + 1.0) / 2.0)
    else:
        score = 0.0
    return score, triggered


def _signal_blending_edges(crops):
    """GAN face-paste seam: border gradient vs interior ratio."""
    if not crops:
        return 0.0, False
    edge_ratios = []
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sobel = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        )
        h, w = sobel.shape
        bw = max(8, int(min(h, w) * 0.10))
        border = np.concatenate([
            sobel[:bw, :].ravel(),
            sobel[-bw:, :].ravel(),
            sobel[:, :bw].ravel(),
            sobel[:, -bw:].ravel(),
        ])
        interior = sobel[bw:-bw, bw:-bw].ravel()
        if interior.size == 0:
            continue
        ratio = float(np.mean(border)) / (float(np.mean(interior)) + 1e-6)
        edge_ratios.append(ratio)
    if not edge_ratios:
        return 0.0, False
    mean_ratio = float(np.mean(edge_ratios))
    deviation  = abs(mean_ratio - 1.0)
    triggered  = deviation > 0.5
    score = min(1.0, deviation / 1.2)
    return score, triggered


def _signal_noise_floor(crops):
    """Gaussian residual noise — GAN faces are too clean."""
    if not crops:
        return 0.0, False
    noise_levels = []
    for crop in crops:
        gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_levels.append(float(np.mean(np.abs(gray - blurred))))
    mean_noise = float(np.mean(noise_levels))
    std_noise  = float(np.std(noise_levels))
    too_clean    = mean_noise < 1.5
    inconsistent = std_noise / (mean_noise + 1e-6) > 0.8
    triggered = too_clean or inconsistent
    if too_clean:
        score = min(1.0, 1.5 / (mean_noise + 0.1))
    elif inconsistent:
        score = min(1.0, std_noise / (mean_noise + 1e-6) / 2.0)
    else:
        score = 0.0
    return score, triggered


def _signal_temporal_flicker(frames):
    """Frame-to-frame brightness inconsistency (video only)."""
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
    """Hue temperature shifts across frames (video only)."""
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
    """Lightweight file-header based metadata score."""
    try:
        size = os.path.getsize(path)
        # Suspiciously small file for its type
        if _is_video(path) and size < 50_000:
            return 0.3, True
        if not _is_video(path) and size < 2_000:
            return 0.2, True
    except Exception:
        pass
    return 0.0, False


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
        return _build_result(0.1, 0.0, 0.0, 0.0, 0.0, ["media_decode_error"], start_time)

    # Face crops
    crops     = _get_face_crops(frames)
    has_faces = len(crops) >= (2 if video else 1)

    # ── Run signals ────────────────────────────────────
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
        tex_score, tex_trig = _signal_face_texture(crops)
        raw["texture"] = tex_score
        if tex_trig:
            signals.append("unnatural_face_texture")

        edge_score, edge_trig = _signal_blending_edges(crops)
        raw["edges"] = edge_score
        if edge_trig:
            signals.append("face_blending_seam")

        noise_score, noise_trig = _signal_noise_floor(crops)
        raw["noise"] = noise_score
        if noise_trig:
            signals.append("abnormal_noise_pattern")
    else:
        signals.append("no_clear_faces_detected")
        raw["texture"] = 0.25
        raw["edges"]   = 0.0
        raw["noise"]   = 0.0

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

    meta_score, meta_trig = _signal_metadata(file_path)
    if meta_trig:
        signals.append("suspicious_metadata_integrity")

    # ── Weighted fusion ────────────────────────────────
    if has_faces:
        model_score    = (
            raw["frequency"] * 0.20 +
            raw["texture"]   * 0.30 +
            raw["edges"]     * 0.25 +
            raw["noise"]     * 0.15 +
            raw["color"]     * 0.10
        )
        artifact_score = raw["compression"]
        temporal_score = raw.get("temporal", 0.0)
    else:
        model_score    = (
            raw["frequency"] * 0.50 +
            raw["texture"]   * 0.10 +
            raw["color"]     * 0.20 +
            raw.get("temporal", 0.0) * 0.20
        )
        artifact_score = raw["compression"]
        temporal_score = raw.get("temporal", 0.0)

    final_score = (
        WEIGHT_MODEL       * model_score    +
        WEIGHT_ARTIFACT    * artifact_score +
        WEIGHT_TEMPORAL    * temporal_score +
        WEIGHT_METADATA    * meta_score     +
        WEIGHT_COMPRESSION * raw["compression"]
    )
    final_score = float(np.clip(final_score, 0.0, 1.0))

    return _build_result(
        model_score, artifact_score, temporal_score,
        meta_score, raw["compression"], signals, start_time, final_score
    )


def _build_result(model_score, artifact_score, temporal_score,
                  metadata_score, compression_score, signals,
                  start_time, final_score=None):
    if final_score is None:
        final_score = (
            WEIGHT_MODEL       * model_score    +
            WEIGHT_ARTIFACT    * artifact_score +
            WEIGHT_TEMPORAL    * temporal_score +
            WEIGHT_METADATA    * metadata_score +
            WEIGHT_COMPRESSION * compression_score
        )
    final_score = float(np.clip(final_score, 0.0, 1.0))

    if final_score >= THRESHOLD_REJECT:
        verdict = "REJECTED"
        if "deepfake_detected" not in signals:
            signals.append("synthetic_generation_signal")
    else:
        verdict = "APPROVED"

    return {
        "model":             "trueframe-signal-analyzer",
        "model_score":       round(float(model_score), 4),
        "artifact_score":    round(float(artifact_score), 4),
        "temporal_score":    round(float(temporal_score), 4),
        "metadata_score":    round(float(metadata_score), 4),
        "compression_score": round(float(compression_score), 4),
        "final_score":       round(final_score, 4),
        "verdict":           verdict,
        "signals":           list(set(signals)),
    }


# ─────────────────── ENTRY POINT ─────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({
            "model":             "trueframe-signal-analyzer",
            "model_score":       0.1,
            "artifact_score":    0.0,
            "temporal_score":    0.0,
            "metadata_score":    0.0,
            "compression_score": 0.0,
            "final_score":       0.1,
            "verdict":           "REJECTED",
            "signals":           ["no_file_provided"],
        }))
        sys.exit(0)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(json.dumps({
            "model":             "trueframe-signal-analyzer",
            "model_score":       0.1,
            "artifact_score":    0.0,
            "temporal_score":    0.0,
            "metadata_score":    0.0,
            "compression_score": 0.0,
            "final_score":       0.1,
            "verdict":           "REJECTED",
            "signals":           [f"file_not_found: {file_path}", "fail_closed"],
        }))
        sys.exit(0)

    try:
        result = analyze(file_path)
        print(json.dumps(result))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({
            "model":             "trueframe-signal-analyzer",
            "model_score":       0.1,
            "artifact_score":    0.0,
            "temporal_score":    0.0,
            "metadata_score":    0.0,
            "compression_score": 0.0,
            "final_score":       0.1,
            "verdict":           "REJECTED",
            "signals":           [f"engine_error: {str(e)}", "fail_closed"],
        }))
        sys.exit(0)
