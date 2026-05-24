"""
TrueFrame — Video Deepfake Detector (Reels)
============================================
Detection strategy: MODEL-FIRST with signal analysis fallback.

Priority 1 — LightFakeDetect ONNX model (MobileNetV2 + CBAM + GRU):
    Loads  ai_service/models/lightfakedetect.onnx  if present.
    Pipeline:
      1. Extract frames from video
      2. SSIM-based similarity filter (remove duplicates — paper's novel step)
      3. MTCNN face detection + crop → 224×224
      4. Normalize (ImageNet mean/std)
      5. Feed frame sequence → GRU → P(fake)

Priority 2 — 10-signal analysis fallback (always available):
    Pure OpenCV + NumPy — no model required.
    Temporal flicker, GAN frequency, eye artifacts, skin tone, etc.

Score fusion (when both available):
    final = 0.70 * model_score + 0.30 * signal_score

Verdict:
    final_score >= 0.80 → REJECTED (deepfake detected)
    final_score <  0.80 → APPROVED (real content)

Usage:
    python -m ai_service.training.reel_inference <video_path>

Output: JSON with scores, verdict, signals, and raw_scores.
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

# ─────────────────── CONFIG ──────────────────────────
THRESHOLD_APPROVE = 0.60   # < 0.60 → APPROVED
THRESHOLD_REJECT  = 0.60   # >= 0.60 → REJECTED

MAX_FRAMES     = 20        # Frames after SSIM deduplication
FRAME_SIZE     = (224, 224)
SSIM_THRESHOLD = 0.95      # Frames more similar than this are duplicates

IMAGENET_MEAN  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD   = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────── VIDEO HELPERS ───────────────────

def _open_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0, 0
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, total


def _extract_raw_frames(video_path, n=MAX_FRAMES * 3):
    """
    Uniformly sample up to n frames from the video.
    Samples 3× the target to give the SSIM filter enough budget.
    """
    cap, fps, total = _open_video(video_path)
    if cap is None:
        return []
    duration = total / max(fps, 1.0)
    times = np.linspace(0.5, max(0.5, duration - 0.5), n)
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
    cap.release()
    return frames


# ─────────────────── SSIM DEDUPLICATION ─────────────

def _ssim_fast(img1, img2):
    """
    Fast SSIM approximation using NumPy.
    Operates on 64×64 grayscale thumbnails for speed.
    Returns value in [0, 1].
    """
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(img1, img2, data_range=255.0))
    except ImportError:
        pass
    # Fallback: luminance + contrast similarity
    mu1, mu2 = img1.mean(), img2.mean()
    s1, s2   = img1.std(), img2.std()
    s12      = float(np.mean((img1 - mu1) * (img2 - mu2)))
    C1, C2   = 6.5025, 58.5225
    num  = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den  = (mu1**2 + mu2**2 + C1) * (s1**2 + s2**2 + C2)
    return float(num / (den + 1e-8))


def _filter_similar_frames(frames, threshold=SSIM_THRESHOLD, max_keep=MAX_FRAMES):
    """
    Remove near-duplicate frames via SSIM comparison.

    This is the key preprocessing innovation from the LightFakeDetect paper:
    instead of randomly dropping frames, we compare consecutive frames and
    only keep frames that are sufficiently different from the previous kept
    frame. This preserves meaningful temporal content.

    Args:
        frames:    List of BGR frames
        threshold: SSIM score above which frame is a duplicate (default 0.95)
        max_keep:  Hard cap on returned frames

    Returns:
        List of informative, non-duplicate frames
    """
    if not frames:
        return []

    thumb = (64, 64)
    kept  = [frames[0]]
    prev  = cv2.resize(
        cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), thumb
    ).astype(np.float32)

    for frame in frames[1:]:
        gray = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), thumb
        ).astype(np.float32)

        if _ssim_fast(prev, gray) < threshold:
            kept.append(frame)
            prev = gray

        if len(kept) >= max_keep:
            break

    return kept


# ─────────────────── FACE DETECTION ──────────────────

def _build_detector():
    """
    Returns detect_fn(frame_bgr) → face_crop_bgr | None.
    Priority: MTCNN (facenet-pytorch) → MediaPipe → Haar cascade.
    """
    # ── MTCNN ──────────────────────────────────────────
    try:
        from facenet_pytorch import MTCNN
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mtcnn = MTCNN(
            image_size=224, margin=20, min_face_size=40,
            keep_all=False, post_process=False, device=device,
        )

        def _mtcnn(frame_bgr):
            from PIL import Image
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            try:
                face_t = mtcnn(pil)
                if face_t is None:
                    return None
                face_np = face_t.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                return cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
            except Exception:
                return None

        logger.debug("Face detector: MTCNN (facenet-pytorch)")
        return _mtcnn
    except ImportError:
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

        logger.debug("Face detector: MediaPipe")
        return _mediapipe
    except ImportError:
        pass

    # ── Haar cascade ───────────────────────────────────
    def _haar(frame_bgr):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        for nb in [3, 2]:
            faces = cascade.detectMultiScale(gray, 1.05, nb, minSize=(32, 32))
            if len(faces) > 0:
                x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                pad = int(min(w, h) * 0.15)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame_bgr.shape[1], x + w + pad)
                y2 = min(frame_bgr.shape[0], y + h + pad)
                return frame_bgr[y1:y2, x1:x2]
        return None

    logger.debug("Face detector: Haar cascade (fallback)")
    return _haar


_DETECTOR = None

def _get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = _build_detector()
    return _DETECTOR


def _extract_face_crops(frames):
    """
    Detect and crop faces from each frame.
    Tries brightened version on failure (handles dark frames).
    Returns list of (224, 224, 3) BGR crops.
    """
    detector = _get_detector()
    crops = []
    for frame in frames:
        face = detector(frame)
        if face is None:
            bright = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)
            face = detector(bright)
        if face is not None and face.size > 0:
            crops.append(cv2.resize(face, FRAME_SIZE))
    return crops


# ─────────────────── NORMALIZATION ───────────────────

def _normalize_crop(face_bgr):
    """BGR uint8 (224×224) → float32 (3, 224, 224) in ImageNet stats."""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return normalized.transpose(2, 0, 1)   # (3, 224, 224)


# ─────────────────── ONNX INFERENCE ──────────────────

_ONNX_SESSION = None

def _get_onnx_session():
    """Load and cache the LightFakeDetect ONNX session. Returns None if unavailable."""
    global _ONNX_SESSION
    if _ONNX_SESSION is not None:
        return _ONNX_SESSION
    try:
        import onnxruntime as ort
        model_dir  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        onnx_path  = os.path.join(model_dir, "lightfakedetect.onnx")
        if not os.path.exists(onnx_path):
            return None
        sess = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _ONNX_SESSION = sess
        logger.info(f"[LightFakeDetect] ONNX loaded: {onnx_path}")
        return sess
    except Exception as e:
        logger.warning(f"[LightFakeDetect] ONNX unavailable: {e}")
        return None


def _run_onnx_inference(face_crops):
    """
    Run LightFakeDetect GRU inference on a sequence of face crops.

    Args:
        face_crops: List of (224, 224, 3) BGR uint8 face crops

    Returns:
        float P(fake) in [0, 1], or None if model unavailable / no faces
    """
    sess = _get_onnx_session()
    if sess is None or not face_crops:
        return None

    try:
        normalized = [_normalize_crop(c) for c in face_crops]  # list of (3, 224, 224)
        seq = np.stack(normalized, axis=0)[np.newaxis].astype(np.float32)
        # seq shape: (1, T, 3, 224, 224)

        input_name = sess.get_inputs()[0].name
        output     = sess.run(None, {input_name: seq})
        prob       = float(output[0][0])
        return float(np.clip(prob, 0.0, 1.0))
    except Exception as e:
        logger.warning(f"[LightFakeDetect] inference error: {e}")
        return None


# ─────────────────── SIGNAL DETECTORS (fallback) ─────

def signal_temporal_flicker(frames):
    """Frame-to-frame brightness inconsistency."""
    if len(frames) < 4:
        return 0.0, False
    brightnesses = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() for f in frames]
    diffs = np.abs(np.diff(brightnesses))
    mean_diff = float(np.mean(diffs))
    max_diff  = float(np.max(diffs))
    score = min(1.0, (mean_diff / 6.0) * 0.4 + (max_diff / 30.0) * 0.6)
    triggered = mean_diff > 5.0 or max_diff > 20.0
    return score, triggered


def signal_block_artifacts(frames):
    """8×8 DCT compression grid artifacts in re-encoded deepfakes."""
    if not frames:
        return 0.0, False
    scores = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        cols = gray[:, :w - (w % 8)]
        row_means = cols.reshape(h, -1, 8).mean(axis=2)
        block_var = float(np.var(row_means))
        pixel_var = float(np.var(gray))
        scores.append(block_var / (pixel_var + 1e-6))
    mean_ratio = float(np.mean(scores))
    score = min(1.0, max(0.0, (mean_ratio - 0.35) / 0.40))
    triggered = mean_ratio > 0.45
    return score, triggered


def signal_color_consistency(frames):
    """Abrupt hue/white-balance shifts between frames."""
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


def signal_face_texture_variance(crops):
    """Unnatural smoothness or sharpness via Laplacian variance."""
    if not crops:
        return 0.0, False
    variances = []
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap  = cv2.Laplacian(gray, cv2.CV_64F)
        variances.append(float(np.var(lap)))
    mean_var = float(np.mean(variances))
    std_var  = float(np.std(variances))
    too_smooth   = mean_var < 30.0
    too_sharp    = mean_var > 15000.0
    unstable_var = std_var / (mean_var + 1.0) > 2.5
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
    """Gradient seam at face-swap boundary."""
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
        bw = max(8, int(min(h, w) * 0.1))
        border = np.concatenate([
            sobel[:bw, :].ravel(), sobel[-bw:, :].ravel(),
            sobel[:, :bw].ravel(), sobel[:, -bw:].ravel(),
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
    score      = min(1.0, deviation / 1.2)
    return score, triggered


def signal_noise_floor(crops):
    """Abnormally clean noise — GAN faces lack camera sensor noise."""
    if not crops:
        return 0.0, False
    noise_levels = []
    for crop in crops:
        gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_levels.append(float(np.mean(np.abs(gray - blurred))))
    mean_noise = float(np.mean(noise_levels))
    std_noise  = float(np.std(noise_levels))
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


def signal_face_gan_frequency(crops):
    """GAN spectral fingerprint in face FFT — high-frequency grid pattern."""
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


def signal_skin_tone_consistency(crops):
    """Cross-frame skin hue instability — GAN face-swaps flicker in color."""
    if len(crops) < 4:
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
    hue_score = min(1.0, max(0.0, (hue_std - 3.0) / 8.0))
    sat_score = min(1.0, max(0.0, (sat_std - 12.0) / 20.0))
    score = 0.6 * hue_score + 0.4 * sat_score
    triggered = hue_std > 4.5 or sat_std > 18.0
    return float(score), triggered


def signal_eye_region_artifacts(crops):
    """Eye-band channel correlation + edge entropy — GANs struggle most here."""
    if len(crops) < 3:
        return 0.0, False
    channel_corrs  = []
    edge_entropies = []
    for crop in crops:
        h   = crop.shape[0]
        eye = crop[int(h * 0.25): int(h * 0.52), :]
        if eye.size == 0:
            continue
        b = eye[:, :, 0].astype(np.float32).ravel()
        g = eye[:, :, 1].astype(np.float32).ravel()
        r = eye[:, :, 2].astype(np.float32).ravel()
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
        hist = hist + 1e-9
        hist = hist / hist.sum()
        entropy = float(-np.sum(hist * np.log2(hist)))
        edge_entropies.append(entropy)
    if not channel_corrs:
        return 0.0, False
    mean_corr    = float(np.mean(channel_corrs))
    corr_score   = min(1.0, max(0.0, (0.80 - mean_corr) / 0.30))
    mean_entropy = float(np.mean(edge_entropies)) if edge_entropies else 4.0
    ent_score    = min(1.0, max(0.0, abs(mean_entropy - 4.0) / 1.5))
    score        = 0.65 * corr_score + 0.35 * ent_score
    triggered    = mean_corr < 0.72 or abs(mean_entropy - 4.0) > 1.2
    return float(score), triggered


def signal_face_color_channel_decoupling(crops):
    """R/G/B inter-channel correlation — real cameras couple channels tightly."""
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
        corr_scores.append(min(rg, rb, gb))
    if not corr_scores:
        return 0.0, False
    mean_min_corr = float(np.mean(corr_scores))
    score    = min(1.0, max(0.0, (0.85 - mean_min_corr) / 0.30))
    triggered = mean_min_corr < 0.75
    return score, triggered


# ─────────────────── SIGNAL ENGINE ───────────────────

def _run_signal_analysis(frames, crops):
    """
    Run all 10 signal detectors on the video.
    Returns (signal_score, signals_list, raw_scores_dict).
    """
    signals    = []
    raw_scores = {}

    # Strict Fail-Closed Check: If fewer than 2 face crops are detected, return 1.0 immediately (fail closed)
    if len(crops) < 2:
        signals.append("no_face_detected")
        signals.append("fail_closed")
        # Populate all raw scores with 0.0 to prevent KeyError in result builder
        for k in ["temporal_flicker", "block_artifacts", "color_shift",
                  "face_texture", "blending_edges", "noise_floor",
                  "gan_frequency", "skin_tone", "eye_artifacts", "channel_decoupling"]:
            raw_scores[k] = 0.0
        return 1.0, signals, raw_scores

    has_faces = len(crops) >= 2

    # Global signals (whole-frame)
    flicker_score, flicker_trig = signal_temporal_flicker(frames)
    raw_scores["temporal_flicker"] = flicker_score
    if flicker_trig:
        signals.append("temporal_flicker_detected")

    block_score, block_trig = signal_block_artifacts(frames)
    raw_scores["block_artifacts"] = block_score
    if block_trig:
        signals.append("compression_block_artifacts")

    color_score, color_trig = signal_color_consistency(frames)
    raw_scores["color_shift"] = color_score
    if color_trig:
        signals.append("color_temperature_inconsistency")

    # Face-specific signals
    if has_faces:
        texture_score, texture_trig = signal_face_texture_variance(crops)
        raw_scores["face_texture"] = texture_score
        if texture_trig:
            signals.append("unnatural_face_texture")

        edge_score, edge_trig = signal_blending_edges(crops)
        raw_scores["blending_edges"] = edge_score
        if edge_trig:
            signals.append("face_blending_seam")

        noise_score, noise_trig = signal_noise_floor(crops)
        raw_scores["noise_floor"] = noise_score
        if noise_trig:
            signals.append("abnormal_noise_pattern")

        gan_freq_score, gan_freq_trig = signal_face_gan_frequency(crops)
        raw_scores["gan_frequency"] = gan_freq_score
        if gan_freq_trig:
            signals.append("gan_spectral_fingerprint")

        skin_score, skin_trig = signal_skin_tone_consistency(crops)
        raw_scores["skin_tone"] = skin_score
        if skin_trig:
            signals.append("skin_tone_instability")

        eye_score, eye_trig = signal_eye_region_artifacts(crops)
        raw_scores["eye_artifacts"] = eye_score
        if eye_trig:
            signals.append("eye_region_gan_artifact")

        channel_score, channel_trig = signal_face_color_channel_decoupling(crops)
        raw_scores["channel_decoupling"] = channel_score
        if channel_trig:
            signals.append("color_channel_decoupled")

        weights = {
            "temporal_flicker": 0.06,
            "block_artifacts":  0.12,
            "color_shift":      0.06,
            "face_texture":     0.10,
            "blending_edges":   0.10,
            "noise_floor":      0.08,
            "gan_frequency":    0.18,
            "skin_tone":        0.10,
            "eye_artifacts":    0.12,
            "channel_decoupling": 0.08,
        }
    else:
        signals.append("no_face_detected")
        for k in ["face_texture", "blending_edges", "noise_floor",
                  "gan_frequency", "skin_tone", "eye_artifacts", "channel_decoupling"]:
            raw_scores[k] = 0.0
        weights = {
            "temporal_flicker": 0.30,
            "block_artifacts":  0.40,
            "color_shift":      0.20,
            "face_texture":     0.03,
            "blending_edges":   0.03,
            "noise_floor":      0.02,
            "gan_frequency":    0.00,
            "skin_tone":        0.00,
            "eye_artifacts":    0.01,
            "channel_decoupling": 0.01,
        }

    signal_score = float(sum(
        raw_scores.get(k, 0.0) * w for k, w in weights.items()
    ))
    signal_score = round(min(1.0, max(0.0, signal_score)), 4)

    return signal_score, signals, raw_scores


# ─────────────────── MAIN ENGINE ─────────────────────

def analyze_video(video_path):
    """
    Analyze a video for deepfake content using LightFakeDetect + signal fallback.

    Returns JSON-compatible dict with scores, verdict, signals, raw_scores.
    """
    start_time = time.time()
    signals    = []

    # ── 1. Extract raw frames ─────────────────────────
    raw_frames = _extract_raw_frames(video_path, n=MAX_FRAMES * 3)
    if len(raw_frames) < 3:
        return _build_result(1.0, ["insufficient_frames", "fail_closed"], start_time)

    # ── 2. SSIM-based deduplication ───────────────────
    unique_frames = _filter_similar_frames(raw_frames, SSIM_THRESHOLD, MAX_FRAMES * 2)
    if not unique_frames:
        unique_frames = raw_frames[:MAX_FRAMES]

    # ── 3. Face detection and crop ────────────────────
    face_crops = _extract_face_crops(unique_frames)

    # ── 4. LightFakeDetect ONNX inference ─────────────
    onnx_score = None
    if face_crops:
        onnx_score = _run_onnx_inference(face_crops)
        if onnx_score is not None:
            logger.info(f"[LightFakeDetect] P(fake)={onnx_score:.4f} "
                        f"from {len(face_crops)} face crops")
            signals.append("lightfakedetect_model_used")

    # ── 5. Signal analysis (fallback / supplement) ────
    signal_score, signal_signals, raw_scores = _run_signal_analysis(
        unique_frames, face_crops
    )
    signals.extend(signal_signals)

    # ── 6. Score fusion ───────────────────────────────
    if onnx_score is not None:
        final_score = 0.70 * onnx_score + 0.30 * signal_score
        logger.info(f"[LightFakeDetect] fused={final_score:.4f} "
                    f"(model={onnx_score:.4f}, signals={signal_score:.4f})")
    else:
        final_score = signal_score
        signals.append("signal_analysis_fallback")
        logger.info(f"[SignalAnalysis] fallback score={final_score:.4f}")

    # ── Deepfake Signal Boosting ───────────────────────
    # Boost final fused score when high-confidence anomalies are detected
    boost = 0.0
    if "gan_spectral_fingerprint" in signals:
        boost += 0.35
    if "face_blending_seam" in signals:
        boost += 0.30
    if "eye_region_gan_artifact" in signals:
        boost += 0.30
    if "unnatural_face_texture" in signals:
        boost += 0.25
    if "color_channel_decoupled" in signals:
        boost += 0.25

    final_score = round(float(np.clip(final_score + boost, 0.0, 1.0)), 4)

    return _build_result(final_score, signals, start_time, raw_scores, onnx_score)


# ─────────────────── RESULT BUILDER ──────────────────

def _build_result(prob, signals, start_time, raw_scores=None, onnx_score=None):
    elapsed_ms   = (time.time() - start_time) * 1000
    prob         = round(float(np.clip(prob, 0, 1)), 4)
    authenticity = round(1.0 - prob, 4)

    if prob >= THRESHOLD_REJECT:
        verdict    = "REJECTED"
        confidence = "HIGH"
        if "deepfake_detected" not in signals:
            signals.append("deepfake_detected")
    else:
        verdict    = "APPROVED"
        confidence = "HIGH"
        signals    = [s for s in signals if s not in ("deepfake_detected",)]

    result = {
        "model":                "lightfakedetect-video-v2",
        "model_score":          prob,
        "artifact_score":       round(float((raw_scores or {}).get("block_artifacts", 0.0)), 4),
        "temporal_score":       round(float((raw_scores or {}).get("temporal_flicker", 0.0)), 4),
        "metadata_score":       0.0,
        "compression_score":    round(float((raw_scores or {}).get("block_artifacts", 0.0)), 4),
        "final_score":          prob,
        "deepfake_probability": prob,
        "authenticity_score":   authenticity,
        "verdict":              verdict,
        "confidence":           confidence,
        "inference_time_ms":    round(elapsed_ms, 1),
        "signals":              list(set(signals)),
        "raw_scores":           {k: round(float(v), 4) for k, v in (raw_scores or {}).items()},
    }
    if onnx_score is not None:
        result["onnx_model_score"] = round(float(onnx_score), 4)

    return result


# ─────────────────── CLI ENTRY POINT ─────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    if len(sys.argv) < 2:
        print(json.dumps({
            "model": "lightfakedetect-video-v2",
            "model_score": 0.5, "artifact_score": 0.0,
            "temporal_score": 0.0, "metadata_score": 0.0,
            "compression_score": 0.0, "final_score": 0.5,
            "deepfake_probability": 0.5, "authenticity_score": 0.5,
            "verdict": "REJECTED", "signals": ["no_file_provided"],
        }))
        sys.exit(1)

    try:
        result = analyze_video(sys.argv[1])
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({
            "model": "lightfakedetect-video-v2",
            "model_score": 0.5, "artifact_score": 0.0,
            "temporal_score": 0.0, "metadata_score": 0.0,
            "compression_score": 0.0, "final_score": 0.5,
            "deepfake_probability": 0.5, "authenticity_score": 0.5,
            "verdict": "REJECTED",
            "signals": [f"engine_error: {str(e)}", "fail_closed"],
        }))
        sys.exit(1)
