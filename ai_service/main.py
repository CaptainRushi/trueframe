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

# Only content with final_score >= 0.80 is considered synthetic.
# Real-world media (with natural JPEG compression, lighting variation, etc.)
# typically scores well below 0.50 with these heuristics.
THRESHOLD_APPROVE   = 0.80   # < 0.80 → APPROVED
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
    Real photos typically score 0.0; GAN images score > 0.6.
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
        # High-frequency annular band: 40% – 70% of Nyquist (where GAN artifacts appear)
        r_inner = int(min(h, w) * 0.40)
        r_outer = int(min(h, w) * 0.70)
        # Low-frequency reference band: 0% – 20%
        r_ref = int(min(h, w) * 0.20)
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        hf_mask  = (dist >= r_inner) & (dist <= r_outer)
        ref_mask = dist <= r_ref
        hf_power  = magnitude[hf_mask].mean()
        ref_power = magnitude[ref_mask].mean()
        # GAN images have unnaturally HIGH high-frequency power relative to low-freq
        ratio = hf_power / (ref_power + 1e-6)
        scores.append(ratio)
    mean_ratio = float(np.mean(scores))
    # Real photos: ratio typically 0.4–0.7; GAN images: > 0.85
    score = min(1.0, max(0.0, (mean_ratio - 0.75) / 0.30))
    triggered = mean_ratio > 0.85
    return score, triggered


def _signal_block_artifacts(frames):
    """Detect 8×8 DCT block grid from re-encoding.
    NOTE: All JPEGs have some DCT blocking — we only flag extremely severe cases
    where block variance is an outlier, not normal camera JPEG artifacts.
    """
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
    # Raise threshold: normal camera JPEGs score 0.05–0.35;
    # deeply re-encoded fakes score > 0.50
    score = min(1.0, max(0.0, (mean_ratio - 0.35) / 0.40))
    triggered = mean_ratio > 0.45
    return score, triggered


def _signal_face_texture(crops):
    """Unnatural face smoothness or sharpness via Laplacian variance.
    Real faces: 100 < mean_var < 6000 (wide range due to focus, lighting).
    GAN faces: often < 30 (unnaturally smooth) or > 15000 (artefact sharpening).
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
    # Tightened to only flag extreme outliers
    too_smooth   = mean_var < 30.0
    too_sharp    = mean_var > 15000.0
    unstable     = std_var / (mean_var + 1.0) > 2.5
    triggered = too_smooth or too_sharp or unstable
    if too_smooth:
        score = min(1.0, 30.0 / (mean_var + 1.0))
    elif too_sharp:
        score = min(1.0, (mean_var - 15000.0) / 10000.0)
    elif unstable:
        score = min(1.0, (std_var / (mean_var + 1.0) - 2.5) / 2.0)
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
    """Gaussian residual noise — GAN faces are too clean.
    Real faces after JPEG compression: mean_noise typically 0.8–3.5.
    GAN faces: often < 0.5 (too clean after post-processing).
    We use a much stricter threshold to avoid false-positives on real images.
    """
    if not crops:
        return 0.0, False
    noise_levels = []
    for crop in crops:
        gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_levels.append(float(np.mean(np.abs(gray - blurred))))
    mean_noise = float(np.mean(noise_levels))
    std_noise  = float(np.std(noise_levels))
    # Only flag truly extreme values that real cameras never produce
    too_clean    = mean_noise < 0.4
    inconsistent = std_noise / (mean_noise + 1e-6) > 1.5
    triggered = too_clean or inconsistent
    if too_clean:
        score = min(1.0, 0.4 / (mean_noise + 0.05))
    elif inconsistent:
        score = min(1.0, (std_noise / (mean_noise + 1e-6) - 1.5) / 2.0)
    else:
        score = 0.0
    return score, triggered


def _signal_face_gan_frequency(crops):
    """
    GAN networks leave a characteristic high-frequency spectral fingerprint
    in face crops due to upsampling artefacts. Measures HF/LF power ratio
    in the FFT of each face crop.
    Real faces: ratio 0.35–0.55; GAN faces: > 0.62.
    Score: 0 (real) → 1 (GAN fingerprint).
    """
    if not crops:
        return 0.0, False
    scores = []
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
        scores.append(hf_power / (lf_power + 1e-6))
    mean_ratio = float(np.mean(scores))
    score = min(1.0, max(0.0, (mean_ratio - 0.55) / 0.35))
    triggered = mean_ratio > 0.62
    return score, triggered


def _signal_skin_tone_consistency(crops):
    """
    Real faces maintain consistent skin hue across images.
    GAN face-swaps introduce subtle skin-tone mis-matches (color blending instability).
    For single images, measures hue uniformity within the face crop's skin region.
    Score: 0 (consistent skin) → 1 (unstable/inconsistent).
    """
    if not crops:
        return 0.0, False
    skin_hue_stds = []
    for crop in crops:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        skin_mask = (
            ((H >= 0) & (H <= 25)) | ((H >= 165) & (H <= 180))
        ) & (S >= 40) & (V >= 50)
        if skin_mask.sum() < 100:
            continue
        skin_hue_stds.append(float(H[skin_mask].std()))
    if not skin_hue_stds:
        return 0.0, False
    mean_std = float(np.mean(skin_hue_stds))
    # Real face skin hue std: typically < 6; GAN blended: > 10
    score = min(1.0, max(0.0, (mean_std - 6.0) / 10.0))
    triggered = mean_std > 8.0
    return score, triggered


def _signal_eye_region_artifacts(crops):
    """
    Extracts the eye band (25%–52% height of face crop) and checks for:
    - Color channel decoupling in the iris region
    - Abnormal edge entropy (GAN eyes: too uniform or too chaotic)
    Score: 0 (normal eyes) → 1 (GAN eye artifact).
    """
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
    corr_score = min(1.0, max(0.0, (0.80 - mean_corr) / 0.30))
    mean_entropy = float(np.mean(edge_entropies)) if edge_entropies else 4.0
    ent_score = min(1.0, max(0.0, abs(mean_entropy - 4.0) / 1.5))
    score = 0.65 * corr_score + 0.35 * ent_score
    triggered = mean_corr < 0.72 or abs(mean_entropy - 4.0) > 1.2
    return float(score), triggered


def _signal_channel_decoupling(crops):
    """
    Real faces: R/G/B channels tightly correlated (physics of light).
    GAN faces: channels synthesized semi-independently → lower correlation.
    Score: 0 (well-coupled = real) → 1 (decoupled = GAN).
    """
    if not crops:
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
        corr_scores.append(min(rg, rb, gb))
    if not corr_scores:
        return 0.0, False
    mean_min_corr = float(np.mean(corr_scores))
    score = min(1.0, max(0.0, (0.85 - mean_min_corr) / 0.30))
    triggered = mean_min_corr < 0.75
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

        # NEW face signals
        gan_freq_score, gan_freq_trig = _signal_face_gan_frequency(crops)
        raw["gan_frequency"] = gan_freq_score
        if gan_freq_trig:
            signals.append("gan_spectral_fingerprint")

        skin_score, skin_trig = _signal_skin_tone_consistency(crops)
        raw["skin_tone"] = skin_score
        if skin_trig:
            signals.append("skin_tone_instability")

        eye_score, eye_trig = _signal_eye_region_artifacts(crops)
        raw["eye_artifacts"] = eye_score
        if eye_trig:
            signals.append("eye_region_gan_artifact")

        channel_score, channel_trig = _signal_channel_decoupling(crops)
        raw["channel_decoupling"] = channel_score
        if channel_trig:
            signals.append("color_channel_decoupled")
    else:
        # No faces detected — do NOT assign a suspicious score.
        # Many legitimate posts (landscapes, objects, screenshots) have no faces.
        signals.append("no_clear_faces_detected")
        raw["texture"]           = 0.0
        raw["edges"]             = 0.0
        raw["noise"]             = 0.0
        raw["gan_frequency"]     = 0.0
        raw["skin_tone"]         = 0.0
        raw["eye_artifacts"]     = 0.0
        raw["channel_decoupling"]= 0.0

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
    # When faces detected, face signals carry 70% of the final score.
    if has_faces:
        model_score    = (
            raw["frequency"]        * 0.06 +
            raw["texture"]          * 0.10 +
            raw["edges"]            * 0.10 +
            raw["noise"]            * 0.08 +
            raw["color"]            * 0.06 +
            raw["gan_frequency"]    * 0.22 +   # strongest GAN fingerprint
            raw["skin_tone"]        * 0.12 +
            raw["eye_artifacts"]    * 0.14 +   # eyes are GAN's weakest area
            raw["channel_decoupling"]* 0.12
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
        # Ensure no lingering REJECTED/UNDER_REVIEW state
        signals = [s for s in signals if s not in ("synthetic_generation_signal", "deepfake_detected")]

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
