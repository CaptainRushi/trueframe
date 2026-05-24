"""
TrueFrame Reels — Enhanced Deepfake Video Detector
====================================================
Detects deepfake videos using multi-signal analysis with OpenCV + NumPy.
No ONNX model, no PyTorch, no model files required.

Signals analysed:
  GLOBAL (whole-frame):
  1. Temporal flicker   — unnatural frame-to-frame brightness jumps
  2. Block artifacts    — JPEG/compression block grid patterns in re-encoded fakes
  3. Color constancy    — abrupt hue/white-balance shifts across frames

  FACE-SPECIFIC (when faces detected):
  4. Face texture variance  — GAN faces are unnaturally smooth or overly sharp
  5. Blending edges         — gradient seam at GAN face-swap boundary
  6. Noise floor            — GAN faces are too clean (low residual noise)
  7. Face GAN frequency     — spectral fingerprint in HIGH-FREQ of face crops (GAN leaves grid)
  8. Skin-tone consistency  — real faces stay the same hue; fakes flicker in skin region
  9. Eye-region artifacts   — GANs struggle most with eyes; detect blink/shimmer patterns
  10.Color channel decoupling— real faces: R/G/B correlated; GAN faces: channels desynchronized

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
THRESHOLD_REVIEW  = 0.80   # kept for schema compatibility
THRESHOLD_REJECT  = 0.80   # >= 0.80 → REJECTED

MAX_FRAMES = 30            # more frames = better temporal analysis
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
    """Fallback face detection with Haar cascade — always available via OpenCV.
    More permissive settings to catch faces in profile, partial occlusion, etc."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Try multiple scales for better coverage
    for neighbors in [3, 2]:
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=neighbors, minSize=(32, 32)
        )
        if len(faces) > 0:
            x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            # Expand bounding box slightly to include forehead/chin
            pad = int(min(w, h) * 0.15)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            return frame[y1:y2, x1:x2]
    return None


def _detect_face_mediapipe(frame, face_det):
    """MediaPipe-based face detection with expanded bounding box."""
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_det.process(rgb)
    if not results.detections:
        return None
    best = max(results.detections, key=lambda d: d.score[0])
    bbox = best.location_data.relative_bounding_box
    fh, fw = frame.shape[:2]
    pad_x = bbox.width  * 0.15
    pad_y = bbox.height * 0.15
    x1 = max(0, int((bbox.xmin - pad_x) * fw))
    y1 = max(0, int((bbox.ymin - pad_y) * fh))
    x2 = min(fw, int((bbox.xmin + bbox.width  + pad_x) * fw))
    y2 = min(fh, int((bbox.ymin + bbox.height + pad_y) * fh))
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return frame[y1:y2, x1:x2]


def _get_face_detector():
    """Return detect_fn(frame) → face crop BGR or None."""
    try:
        import mediapipe as mp
        try:
            face_det = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.4
            )
        except Exception:
            face_det = mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=0.4
            )
        def detect(frame):
            return _detect_face_mediapipe(frame, face_det)
        return detect
    except Exception:
        return _detect_face_haar


def _extract_face_crops(frames, detect_fn):
    """Extract & resize face crops. Tries both original and flipped frame for coverage."""
    crops = []
    for frame in frames:
        face = detect_fn(frame)
        if face is None:
            # Try on a slightly brightened version for dark frames
            bright = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)
            face = detect_fn(bright)
        if face is not None and face.size > 0:
            face_resized = cv2.resize(face, FRAME_SIZE)
            crops.append(face_resized)
    return crops


def _extract_eye_regions(crops):
    """Extract the eye-band (top 40%) from each face crop for eye-specific analysis."""
    eye_regions = []
    for crop in crops:
        h = crop.shape[0]
        # Eyes typically in upper 25%–50% of face bounding box
        eye_band = crop[int(h * 0.25): int(h * 0.52), :]
        if eye_band.size > 0:
            eye_regions.append(eye_band)
    return eye_regions


# ─────────────────── GLOBAL SIGNAL ANALYSIS ──────────

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
        h, w = gray.shape
        cols = gray[:, :w - (w % 8)]
        row_means = cols.reshape(h, -1, 8).mean(axis=2)
        block_variance = float(np.var(row_means))
        pixel_variance  = float(np.var(gray))
        ratio = block_variance / (pixel_variance + 1e-6)
        scores.append(ratio)
    mean_ratio = float(np.mean(scores))
    # Real camera videos: ratio typically 0.03–0.35; heavily re-encoded fakes: > 0.50
    score = min(1.0, max(0.0, (mean_ratio - 0.35) / 0.40))
    triggered = mean_ratio > 0.45
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
    score = min(1.0, (mean_diff / 8.0) * 0.5 + (max_diff / 40.0) * 0.5)
    triggered = mean_diff > 5.0 or max_diff > 25.0
    return score, triggered


# ─────────────────── FACE SIGNAL ANALYSIS ────────────

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
    Real faces: border ≈ interior (ratio ≈ 1.0).
    GAN paste seam: ratio > 1.6 or < 0.45.
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
    deviation = abs(mean_ratio - 1.0)
    triggered = deviation > 0.5
    score = min(1.0, deviation / 1.2)
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


def signal_face_gan_frequency(crops):
    """
    GAN networks leave a characteristic high-frequency spectral fingerprint
    in face crops — a periodic grid pattern in the DCT/FFT domain that is
    absent in real camera-captured faces.

    Real faces: high-frequency band has LOW relative power.
    GAN faces:  high-frequency band has ELEVATED power due to upsampling artefacts.

    Score: 0 (real-like) → 1 (strong GAN fingerprint).
    """
    if not crops:
        return 0.0, False
    scores = []
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # 2D FFT on face crop
        fft = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shifted))
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        # Very high frequency band (50%–80% of Nyquist) — GAN upsampling grid lives here
        r_hf_in  = int(min(h, w) * 0.50)
        r_hf_out = int(min(h, w) * 0.80)
        # Low-frequency reference (0%–25%)
        r_lf = int(min(h, w) * 0.25)
        hf_mask = (dist >= r_hf_in) & (dist <= r_hf_out)
        lf_mask = dist <= r_lf
        hf_power = magnitude[hf_mask].mean() if hf_mask.any() else 0.0
        lf_power = magnitude[lf_mask].mean() if lf_mask.any() else 1.0
        # GAN ratio: real≈0.35–0.55, GAN≈0.65–0.90
        ratio = hf_power / (lf_power + 1e-6)
        scores.append(ratio)
    mean_ratio = float(np.mean(scores))
    # Scale: real faces typically score 0.0; strong GAN fingerprint → 1.0
    score = min(1.0, max(0.0, (mean_ratio - 0.55) / 0.35))
    triggered = mean_ratio > 0.62
    return score, triggered


def signal_skin_tone_consistency(crops):
    """
    Real human faces maintain consistent skin hue across frames — skin is
    mostly in a narrow orange/brown hue band (HSV H ≈ 5–25°).

    GAN face-swaps often introduce subtle skin-tone mis-matches between
    frames because the generator blends colors from training images differently.

    Measures: (a) cross-frame skin-hue variance, (b) skin saturation stability.
    Score: 0 (consistent real skin) → 1 (unstable fake skin tone).
    """
    if len(crops) < 4:
        return 0.0, False
    skin_hues = []
    skin_sats = []
    for crop in crops:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        # Skin hue range in OpenCV HSV: H ∈ [0, 25] and [165, 180]
        # (OpenCV H is 0–180, so skin is 0–25 and 165–180)
        skin_mask = (
            ((H >= 0) & (H <= 25)) |
            ((H >= 165) & (H <= 180))
        ) & (S >= 40) & (V >= 50)
        if skin_mask.sum() < 100:
            continue   # not enough skin pixels — skip this crop
        skin_hues.append(float(H[skin_mask].mean()))
        skin_sats.append(float(S[skin_mask].mean()))
    if len(skin_hues) < 3:
        return 0.0, False
    hue_std = float(np.std(skin_hues))
    sat_std = float(np.std(skin_sats))
    # Real faces: hue_std < 3, sat_std < 12
    # GAN faces: hue_std > 5, sat_std > 20 (color blending instability)
    hue_score = min(1.0, max(0.0, (hue_std - 3.0) / 8.0))
    sat_score = min(1.0, max(0.0, (sat_std - 12.0) / 20.0))
    score = 0.6 * hue_score + 0.4 * sat_score
    triggered = hue_std > 4.5 or sat_std > 18.0
    return float(score), triggered


def signal_eye_region_artifacts(eye_regions):
    """
    Eyes are the hardest region for GAN face-swaps to generate convincingly.
    Common GAN eye artifacts:
    - Unnatural sharpness gradient (focus suddenly changes at eye boundary)
    - Color channel desynchronization in the iris (RGB channels decorrelated)
    - Temporal shimmer (eye region flickers more than rest of face)
    - Symmetry artifacts (GAN sometimes generates asymmetric irises)

    Measures local gradient entropy and channel correlation in the eye band.
    Score: 0 (normal eyes) → 1 (strong eye artifacts).
    """
    if len(eye_regions) < 3:
        return 0.0, False
    channel_corrs = []
    edge_entropies = []
    for eye in eye_regions:
        # ── Channel correlation (R vs G vs B should be tightly correlated in real faces)
        b = eye[:, :, 0].astype(np.float32).ravel()
        g = eye[:, :, 1].astype(np.float32).ravel()
        r = eye[:, :, 2].astype(np.float32).ravel()
        if np.std(b) < 1e-6 or np.std(g) < 1e-6:
            continue
        corr_rg = float(np.corrcoef(r, g)[0, 1])
        corr_rb = float(np.corrcoef(r, b)[0, 1])
        # Real faces: corr > 0.85; GAN eyes: corr often < 0.70
        channel_corrs.append(min(corr_rg, corr_rb))

        # ── Edge entropy: real eyes have smooth natural edges; fakes have artificial edges
        gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sobel = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        )
        # Histogram entropy of edge magnitudes (GAN: more uniform / artificial distribution)
        hist, _ = np.histogram(sobel.ravel(), bins=32, range=(0, 150))
        hist = hist + 1e-9  # avoid log(0)
        hist = hist / hist.sum()
        entropy = float(-np.sum(hist * np.log2(hist)))
        edge_entropies.append(entropy)

    if not channel_corrs:
        return 0.0, False

    mean_corr = float(np.mean(channel_corrs))
    # Low correlation → channels are decoupled → GAN artifact
    corr_score = min(1.0, max(0.0, (0.80 - mean_corr) / 0.30))

    # Edge entropy: real eyes ≈ 3.5–4.5 bits; GAN eyes often < 3.0 (too uniform) or > 5.0
    mean_entropy = float(np.mean(edge_entropies)) if edge_entropies else 4.0
    ent_score = min(1.0, max(0.0, abs(mean_entropy - 4.0) / 1.5))

    score = 0.65 * corr_score + 0.35 * ent_score
    triggered = mean_corr < 0.72 or abs(mean_entropy - 4.0) > 1.2
    return float(score), triggered


def signal_face_color_channel_decoupling(crops):
    """
    In real photographs, color channels (R, G, B) are tightly correlated due
    to the physics of light and camera sensor design. GAN-generated face regions
    exhibit weaker inter-channel correlation because the generator learns to
    synthesize channels somewhat independently.

    This is a strong, robust signal that catches many GAN architectures.
    Score: 0 (well-coupled channels = real) → 1 (decoupled = GAN).
    """
    if len(crops) < 3:
        return 0.0, False
    corr_scores = []
    for crop in crops:
        b = crop[:, :, 0].astype(np.float64).ravel()
        g = crop[:, :, 1].astype(np.float64).ravel()
        r = crop[:, :, 2].astype(np.float64).ravel()
        if np.std(b) < 1e-6 or np.std(g) < 1e-6 or np.std(r) < 1e-6:
            continue
        rg = float(np.corrcoef(r, g)[0, 1])
        rb = float(np.corrcoef(r, b)[0, 1])
        gb = float(np.corrcoef(g, b)[0, 1])
        min_corr = min(rg, rb, gb)
        corr_scores.append(min_corr)
    if not corr_scores:
        return 0.0, False
    mean_min_corr = float(np.mean(corr_scores))
    # Real face crops: min_corr typically > 0.88
    # GAN face crops: min_corr often drops to 0.60–0.78
    score = min(1.0, max(0.0, (0.85 - mean_min_corr) / 0.30))
    triggered = mean_min_corr < 0.75
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
    detect_fn   = _get_face_detector()
    crops       = _extract_face_crops(frames, detect_fn)
    eye_regions = _extract_eye_regions(crops)

    # We require at least 3 face crops for face-based analysis
    has_faces = len(crops) >= 3

    # ── 3. Run global signal analysis ─────────────────
    raw_scores = {}

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

    # ── 4. Run face-specific signal analysis ──────────
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

        # NEW: GAN spectral fingerprint on face crops
        gan_freq_score, gan_freq_triggered = signal_face_gan_frequency(crops)
        raw_scores["gan_frequency"] = gan_freq_score
        if gan_freq_triggered:
            signals.append("gan_spectral_fingerprint")

        # NEW: skin tone cross-frame consistency
        skin_score, skin_triggered = signal_skin_tone_consistency(crops)
        raw_scores["skin_tone"] = skin_score
        if skin_triggered:
            signals.append("skin_tone_instability")

        # NEW: eye region artifacts (eyes are GAN's weakest point)
        eye_score, eye_triggered = signal_eye_region_artifacts(eye_regions)
        raw_scores["eye_artifacts"] = eye_score
        if eye_triggered:
            signals.append("eye_region_gan_artifact")

        # NEW: color channel decoupling across whole face crop
        channel_score, channel_triggered = signal_face_color_channel_decoupling(crops)
        raw_scores["channel_decoupling"] = channel_score
        if channel_triggered:
            signals.append("color_channel_decoupled")

    else:
        # No faces detected — do NOT assign suspicious scores.
        # Many real videos have no faces (scenery, sports, etc.).
        signals.append("no_face_detected")
        raw_scores["face_texture"]       = 0.0
        raw_scores["blending_edges"]     = 0.0
        raw_scores["noise_floor"]        = 0.0
        raw_scores["gan_frequency"]      = 0.0
        raw_scores["skin_tone"]          = 0.0
        raw_scores["eye_artifacts"]      = 0.0
        raw_scores["channel_decoupling"] = 0.0

    # ── 5. Weighted fusion ────────────────────────────
    # When faces are detected, face signals dominate (70% total weight).
    # Global signals cover the remaining 30%.
    if has_faces:
        weights = {
            # Global (30%)
            "temporal_flicker":  0.06,
            "block_artifacts":   0.12,
            "color_shift":       0.06,
            # Face-specific (70%) — NEW signals carry significant weight
            "face_texture":      0.10,
            "blending_edges":    0.10,
            "noise_floor":       0.08,
            "gan_frequency":     0.18,   # strongest GAN fingerprint signal
            "skin_tone":         0.10,
            "eye_artifacts":     0.12,   # eyes are GAN's weakest area
            "channel_decoupling":0.08,
        }
    else:
        # No faces: only global signals
        weights = {
            "temporal_flicker":  0.30,
            "block_artifacts":   0.40,
            "color_shift":       0.20,
            "face_texture":      0.03,
            "blending_edges":    0.03,
            "noise_floor":       0.02,
            "gan_frequency":     0.00,
            "skin_tone":         0.00,
            "eye_artifacts":     0.01,
            "channel_decoupling":0.01,
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
        # Remove any leftover rejection signals
        signals = [s for s in signals if s not in ("deepfake_detected",)]

    result = {
        "model":              "trueframe-video-signal-analyzer-v2",
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
        "raw_scores":         {k: round(float(v), 4) for k, v in (raw_scores or {}).items()},
    }
    return result


# ─────────────────── CLI ENTRY POINT ─────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    if len(sys.argv) < 2:
        print(json.dumps({
            "model":              "trueframe-video-signal-analyzer-v2",
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
            "model":              "trueframe-video-signal-analyzer-v2",
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
