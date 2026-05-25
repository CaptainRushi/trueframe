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
import numpy as np
import cv2
from config import THRESHOLD_APPROVE, THRESHOLD_REJECT

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

def _get_onnx_session():
    """
    Load the LightFakeDetect ONNX model if available.
    Returns (session, model_dir) or (None, None).
    """
    try:
        import onnxruntime as ort
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        candidates = [
            os.path.join(model_dir, "lightfakedetect.onnx"),
            os.path.join(model_dir, "trueframe_reels_detector.onnx"),
        ]
        onnx_path = next((path for path in candidates if os.path.exists(path)), None)
        if onnx_path is None:
            return None, None
        sess = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _log(f"[LightFakeDetect] ONNX model loaded from {onnx_path}")
        return sess, model_dir
    except Exception as e:
        _log(f"[LightFakeDetect] ONNX load failed: {e}")
        return None, None


def _normalize_crop(face_bgr):
    """BGR uint8 → float32 (3, 224, 224) normalized to ImageNet stats."""
    face_bgr = cv2.resize(face_bgr, FRAME_SIZE)
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return normalized.transpose(2, 0, 1)   # (3, 224, 224)


_DETECTOR = None

def _build_detector():
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


def _detect_face(frame):
    """Detect and return the largest face crop from a frame using cached detector."""
    detector = _get_detector()
    return detector(frame)


def _run_onnx_inference(sess, frames):
    """
    Run LightFakeDetect ONNX inference on a list of frames.

    Extracts face crops from frames, normalizes, and feeds the sequence
    through the GRU to get P(fake).

    Returns float P(fake) in [0, 1], or None if inference fails.
    """
    try:
        crops = []
        for frame in frames:
            face = _detect_face(frame)
            if face is None:
                bright = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)
                face = _detect_face(bright)
            if face is not None and face.size > 0:
                crops.append(_normalize_crop(face))

        if len(crops) < 1:
            return None   # No faces detected — can't use model

        # Pad or truncate to 10 frames for model's expected temporal dimension
        if len(crops) < 10:
            while len(crops) < 10:
                crops.append(crops[-1] if crops else crops)
        elif len(crops) > 10:
            crops = crops[:10]

        # Stack into (1, 10, 3, 224, 224)
        seq = np.stack(crops, axis=0)[np.newaxis].astype(np.float32)
        input_name = sess.get_inputs()[0].name
        output = sess.run(None, {input_name: seq})
        prob = float(output[0].flatten()[0])
        return float(np.clip(prob, 0.0, 1.0))
    except Exception as e:
        _log(f"[LightFakeDetect] ONNX inference error: {e}")
        return None


def _get_face_crops(frames):
    """Detect and crop faces from each frame using cached detector."""
    crops = []
    detector = _get_detector()
    for frame in frames:
        face = detector(frame)
        if face is None:
            # Try on brightened version for dark frames
            bright = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)
            face = detector(bright)
        if face is not None and face.size > 0:
            crops.append(cv2.resize(face, FRAME_SIZE))
    return crops


def _get_primary_face_track(frames):
    """Return a single face crop per frame (or None) for temporal analysis."""
    if not frames:
        return []
    detector = _get_detector()
    track = []
    for frame in frames:
        face = detector(frame)
        if face is None:
            bright = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)
            face = detector(bright)
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
    score = min(1.0, max(0.0, (mean_ratio - 0.75) / 0.30))
    triggered = mean_ratio > 0.85
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
    score = min(1.0, max(0.0, mean_ratio * 4.0))
    triggered = mean_ratio > 0.20
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
    too_smooth   = mean_var < 80.0
    too_sharp    = mean_var > 15000.0
    unstable     = std_var / (mean_var + 1.0) > 2.0
    triggered = too_smooth or too_sharp or unstable
    if too_smooth:
        score = min(1.0, 80.0 / (mean_var + 1.0))
    elif too_sharp:
        score = min(1.0, (mean_var - 15000.0) / 10000.0)
    elif unstable:
        score = min(1.0, (std_var / (mean_var + 1.0) - 1.8) / 2.0)
    else:
        score = 0.0
    return score, triggered


def _signal_blending_edges(crops):
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
    triggered  = deviation > 0.30
    score = min(1.0, deviation / 0.8)
    return score, triggered


def _signal_noise_floor(crops):
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


def _signal_face_gan_frequency(crops):
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
    triggered = mean_ratio > 0.58
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
    hue_score = min(1.0, max(0.0, (hue_std - 3.0) / 8.0))
    sat_score = min(1.0, max(0.0, (sat_std - 12.0) / 20.0))
    score = 0.6 * hue_score + 0.4 * sat_score
    triggered = hue_std > 4.5 or sat_std > 18.0
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
    corr_score = min(1.0, max(0.0, (0.80 - mean_corr) / 0.30))
    mean_entropy = float(np.mean(edge_entropies)) if edge_entropies else 4.0
    ent_score = min(1.0, max(0.0, abs(mean_entropy - 4.0) / 1.5))
    score = 0.65 * corr_score + 0.35 * ent_score
    triggered = mean_corr < 0.72 or abs(mean_entropy - 4.0) > 1.2
    return float(score), triggered


def _signal_channel_decoupling(crops):
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
        signals.append("no_clear_faces_detected")
        for k in ["texture", "edges", "noise", "gan_frequency",
                  "skin_tone", "eye_artifacts", "channel_decoupling"]:
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

    # Strict Fail-Closed Check: If no faces are detected, score is automatically 1.0 (REJECTED)
    if not has_faces:
        signals.append("fail_closed")
        return 1.0, signals, raw

    if video:
        model_score = (
            raw["frequency"]         * 0.06 +
            raw["texture"]           * 0.10 +
            raw["edges"]             * 0.10 +
            raw["noise"]             * 0.08 +
            raw["color"]             * 0.06 +
            raw["gan_frequency"]     * 0.22 +
            raw["skin_tone"]         * 0.12 +
            raw["eye_artifacts"]     * 0.14 +
            raw["channel_decoupling"]* 0.12
        )
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
        # Image Fallback Math Fix (Renormalized weights to sum to 1.0)
        # We don't have temporal/expression/color signals for images.
        model_score_raw = (
            raw["frequency"]         * 0.06 +
            raw["texture"]           * 0.10 +
            raw["edges"]             * 0.10 +
            raw["noise"]             * 0.08 +
            raw["gan_frequency"]     * 0.22 +
            raw["skin_tone"]         * 0.12 +
            raw["eye_artifacts"]     * 0.14 +
            raw["channel_decoupling"]* 0.12
        )
        model_score = model_score_raw / 0.94  # normalize because color is 0.0
        artifact_score = raw["compression"]

        # Use image-specific normalized weights summing to 1.0
        final_score = (
            0.70 * model_score +
            0.20 * artifact_score +
            0.10 * meta_score
        )

    final_score = float(np.clip(final_score, 0.0, 1.0))

    return final_score, signals, raw


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

    crops     = _get_face_crops(frames)
    has_faces = len(crops) >= (2 if video else 1)

    # ── Priority 1: LightFakeDetect ONNX model ─────────
    onnx_sess, _ = _get_onnx_session()
    onnx_score   = None

    if onnx_sess is not None:
        onnx_score = _run_onnx_inference(onnx_sess, frames)
        if onnx_score is not None:
            _log(f"[LightFakeDetect] model P(fake)={onnx_score:.4f}")
            signals.append("lightfakedetect_model_used")

    # ── Priority 2: Signal analysis (always run) ───────
    signal_score, signal_signals, raw = _run_signal_analysis(
        file_path, frames, crops, has_faces, video
    )
    signals.extend(signal_signals)

    # ── Score fusion ────────────────────────────────────
    if onnx_score is not None:
        # Both available: model-weighted fusion (boost signals when expressions look off)
        signal_weight = 0.30
        model_weight = 0.70
        if video and raw.get("expression", 0.0) >= 0.55:
            signal_weight = 0.45
            model_weight = 0.55
        final_score = model_weight * onnx_score + signal_weight * signal_score
        _log(f"[LightFakeDetect] fused score={final_score:.4f} "
             f"(model={onnx_score:.4f}, signals={signal_score:.4f})")
    else:
        # Fallback to signal analysis only
        final_score = signal_score
        signals.append("signal_analysis_fallback")
        _log(f"[SignalAnalysis] fallback score={final_score:.4f}")

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

    final_score = float(np.clip(final_score + boost, 0.0, 1.0))

    return _build_result(
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
    )


def _build_result(model_score, artifact_score, temporal_score,
                  expression_score, metadata_score, compression_score, signals,
                  start_time, final_score=None, onnx_score=None):
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

    if final_score >= THRESHOLD_REJECT:
        verdict = "REJECTED"
        if "deepfake_detected" not in signals:
            signals.append("synthetic_generation_signal")
    elif final_score >= THRESHOLD_APPROVE:
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
