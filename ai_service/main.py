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
    final_score >= 0.80 → REJECTED (deepfake detected)
    final_score <  0.80 → APPROVED (real content)

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

# ─────────────────── CONFIG ──────────────────────────
THRESHOLD_REJECT    = 0.80   # >= 0.80 → REJECTED
THRESHOLD_APPROVE   = 0.80   # <  0.80 → APPROVED

MAX_FRAMES          = 20
FRAME_SIZE          = (224, 224)
IMAGENET_MEAN       = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD        = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Signal analysis weights (fallback path)
WEIGHT_MODEL        = 0.40
WEIGHT_ARTIFACT     = 0.20
WEIGHT_TEMPORAL     = 0.15
WEIGHT_METADATA     = 0.10
WEIGHT_COMPRESSION  = 0.15


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
        onnx_path = os.path.join(model_dir, "lightfakedetect.onnx")
        if not os.path.exists(onnx_path):
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


def _detect_face(frame):
    """Detect and return the largest face crop from a frame."""
    # Try MTCNN first (facenet-pytorch)
    try:
        from facenet_pytorch import MTCNN
        import torch
        mtcnn = MTCNN(
            image_size=224, margin=20, min_face_size=40,
            keep_all=False, post_process=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        from PIL import Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        face_t = mtcnn(pil)
        if face_t is not None:
            face_np = face_t.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            return cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
    except Exception:
        pass

    # MediaPipe fallback
    try:
        import mediapipe as mp
        det = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.4
        )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = det.process(rgb)
        if res.detections:
            best = max(res.detections, key=lambda d: d.score[0])
            bbox = best.location_data.relative_bounding_box
            fh, fw = frame.shape[:2]
            px = bbox.width * 0.15
            py = bbox.height * 0.15
            x1 = max(0, int((bbox.xmin - px) * fw))
            y1 = max(0, int((bbox.ymin - py) * fh))
            x2 = min(fw, int((bbox.xmin + bbox.width + px) * fw))
            y2 = min(fh, int((bbox.ymin + bbox.height + py) * fh))
            if x2 - x1 >= 32 and y2 - y1 >= 32:
                return frame[y1:y2, x1:x2]
    except Exception:
        pass

    # Haar cascade fallback
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for nb in [3, 2]:
        faces = cascade.detectMultiScale(gray, 1.05, nb, minSize=(32, 32))
        if len(faces) > 0:
            x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            pad = int(min(w, h) * 0.15)
            return frame[max(0, y-pad):min(frame.shape[0], y+h+pad),
                         max(0, x-pad):min(frame.shape[1], x+w+pad)]
    return None


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

        # Stack into (1, T, 3, 224, 224)
        seq = np.stack(crops, axis=0)[np.newaxis].astype(np.float32)
        input_name = sess.get_inputs()[0].name
        output = sess.run(None, {input_name: seq})
        prob = float(output[0][0])
        return float(np.clip(prob, 0.0, 1.0))
    except Exception as e:
        _log(f"[LightFakeDetect] ONNX inference error: {e}")
        return None


# ─────────────────── SIGNAL DETECTORS (fallback) ─────

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
        haar_crops = _detect_faces_haar(frame)
        for c in haar_crops:
            if c.shape[0] >= 32 and c.shape[1] >= 32:
                crops.append(cv2.resize(c, FRAME_SIZE))
    return crops


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
        cols = gray[:, :w - (w % 8)]
        if cols.shape[1] < 8:
            scores.append(0.0)
            continue
        row_means = cols.reshape(h, -1, 8).mean(axis=2)
        block_var = float(np.var(row_means))
        pixel_var = float(np.var(gray))
        scores.append(block_var / (pixel_var + 1e-6))
    mean_ratio = float(np.mean(scores))
    score = min(1.0, max(0.0, (mean_ratio - 0.35) / 0.40))
    triggered = mean_ratio > 0.45
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
    triggered  = deviation > 0.5
    score = min(1.0, deviation / 1.2)
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
    triggered = mean_ratio > 0.62
    return score, triggered


def _signal_skin_tone_consistency(crops):
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
    score = min(1.0, max(0.0, (mean_std - 6.0) / 10.0))
    triggered = mean_std > 8.0
    return score, triggered


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

    meta_score, meta_trig = _signal_metadata(file_path)
    if meta_trig:
        signals.append("suspicious_metadata_integrity")

    if has_faces:
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
    else:
        model_score = (
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
        return _build_result(0.1, 0.0, 0.0, 0.0, 0.0,
                             ["media_decode_error"], start_time)

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
        # Both available: model-weighted fusion
        final_score = 0.70 * onnx_score + 0.30 * signal_score
        _log(f"[LightFakeDetect] fused score={final_score:.4f} "
             f"(model={onnx_score:.4f}, signals={signal_score:.4f})")
    else:
        # Fallback to signal analysis only
        final_score = signal_score
        signals.append("signal_analysis_fallback")
        _log(f"[SignalAnalysis] fallback score={final_score:.4f}")

    final_score = float(np.clip(final_score, 0.0, 1.0))

    return _build_result(
        raw.get("frequency", 0.0),
        raw["compression"],
        raw.get("temporal", 0.0),
        0.0,
        raw["compression"],
        signals,
        start_time,
        final_score,
        onnx_score,
    )


def _build_result(model_score, artifact_score, temporal_score,
                  metadata_score, compression_score, signals,
                  start_time, final_score=None, onnx_score=None):
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
        signals = [s for s in signals if s not in
                   ("synthetic_generation_signal", "deepfake_detected")]

    result = {
        "model":             "lightfakedetect-v2",
        "model_score":       round(float(model_score), 4),
        "artifact_score":    round(float(artifact_score), 4),
        "temporal_score":    round(float(temporal_score), 4),
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
            "temporal_score": 0.0, "metadata_score": 0.0,
            "compression_score": 0.0, "final_score": 0.1,
            "verdict": "REJECTED", "signals": ["no_file_provided"],
        }))
        sys.exit(0)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(json.dumps({
            "model": "lightfakedetect-v2",
            "model_score": 0.1, "artifact_score": 0.0,
            "temporal_score": 0.0, "metadata_score": 0.0,
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
            "temporal_score": 0.0, "metadata_score": 0.0,
            "compression_score": 0.0, "final_score": 0.1,
            "verdict": "REJECTED",
            "signals": [f"engine_error: {str(e)}", "fail_closed"],
        }))
        sys.exit(0)
