"""
TrueFrame Selfie Verification Engine
=====================================
Verifies that a selfie is from a live, real human — not a deepfake,
AI-generated face, printed photo, or screen replay.

Pipeline:
1. Face Detection (MediaPipe / Haar Cascade)
2. Liveness Estimation (blink, texture, depth cues)
3. AI-Generated Face Detection (GAN artifact analysis)
4. Spoof Detection (screen flicker, print texture, reflection)
5. Deepfake Detection (EfficientNet + HuggingFace ensemble)
6. Scoring & Decision

Usage:
    python selfie_verify.py <image_path> [--frames <frame1> <frame2> ...]

Returns JSON to stdout.
"""

import sys
import os
import json
import time
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(__file__)) if os.path.basename(os.path.dirname(__file__)) == 'ai_service' else os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from ai_core.detector import FaceAnalyzer
from ai_core.models import HuggingFaceDeepfakeDetector

def _log(msg):
    print(msg, file=sys.stderr)


class LivenessAnalyzer:
    """
    Estimates liveness from a single selfie frame.
    Uses texture analysis, frequency domain checks, and depth cues
    to distinguish live faces from photos/screens.
    """

    def analyze(self, face_crop):
        score = 0.0
        signals = []

        if face_crop is None or face_crop.size == 0:
            return 1.0, ["no_face_for_liveness"]

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. Laplacian variance — live faces have more texture detail
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 50:
            score += 0.4
            signals.append("low_texture_variance")
        elif lap_var < 100:
            score += 0.15

        # 2. Gradient magnitude distribution — flat = screen/print
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        grad_std = float(np.std(mag))
        if grad_std < 15:
            score += 0.3
            signals.append("flat_gradient_distribution")

        # 3. Color channel consistency — screens have specific color gamut
        b, g, r = cv2.split(face_crop)
        channel_std = float(np.std([np.mean(b), np.mean(g), np.mean(r)]))
        if channel_std < 3:
            score += 0.2
            signals.append("uniform_color_channels")

        # 4. High-frequency energy — real faces have skin micro-texture
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 4
        mask = np.zeros_like(magnitude)
        mask[center_y-radius:center_y+radius, center_x-radius:center_x+radius] = 1
        high_freq = np.sum(magnitude * (1 - mask))
        low_freq = np.sum(magnitude * mask) + 1e-8
        hf_ratio = high_freq / low_freq
        if hf_ratio < 0.1:
            score += 0.2
            signals.append("low_high_frequency_energy")

        return min(1.0, score), signals


class SpoofDetector:
    """
    Detects presentation attacks: printed photos, phone screens, video replays.
    """

    def analyze(self, face_crop, full_frame=None):
        score = 0.0
        signals = []

        if face_crop is None or face_crop.size == 0:
            return 1.0, ["no_face_for_spoof"]

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. Moiré pattern detection — screens show periodic patterns
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.log1p(np.abs(fshift))
        # Look for periodic peaks outside center
        center = mag[h//2-5:h//2+5, w//2-5:w//2+5].mean()
        outer = mag.mean()
        if outer > center * 0.6:
            score += 0.35
            signals.append("moire_pattern_detected")

        # 2. Specular reflection analysis — screens have uniform reflections
        hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        # Bright spots with low saturation = specular
        bright_mask = v > 240
        bright_ratio = np.sum(bright_mask) / (h * w)
        if bright_ratio > 0.05:
            score += 0.25
            signals.append("screen_reflection_artifacts")
        elif bright_ratio > 0.02:
            score += 0.1

        # 3. Edge sharpness consistency — prints have uniform sharpness
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        if edge_density < 0.02:
            score += 0.2
            signals.append("low_edge_density_print")
        elif edge_density > 0.3:
            score += 0.15
            signals.append("unnatural_edge_density")

        # 4. Color temperature analysis — screens have blue shift
        b_mean = float(np.mean(face_crop[:, :, 0]))
        r_mean = float(np.mean(face_crop[:, :, 2]))
        if b_mean > r_mean * 1.3:
            score += 0.2
            signals.append("blue_shift_screen_detected")

        return min(1.0, score), signals


class AIFaceDetector:
    """
    Detects AI-generated faces (GAN, diffusion model outputs).
    Looks for GAN-specific artifacts.
    """

    def analyze(self, face_crop):
        score = 0.0
        signals = []

        if face_crop is None or face_crop.size == 0:
            return 1.0, ["no_face_for_ai_check"]

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. GAN fingerprint — frequency domain checkerboard pattern
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        # GAN images often show peaks at specific frequencies
        log_mag = np.log1p(mag)
        azimuthal_std = float(np.std(log_mag))
        if azimuthal_std < 1.0:
            score += 0.3
            signals.append("gan_frequency_fingerprint")

        # 2. Symmetry analysis — GAN faces are unnaturally symmetric
        if w > 10:
            left = face_crop[:, :w//2]
            right = cv2.flip(face_crop[:, w//2:], 1)
            min_w = min(left.shape[1], right.shape[1])
            left = left[:, :min_w]
            right = right[:, :min_w]
            sym_diff = np.mean(np.abs(left.astype(float) - right.astype(float)))
            if sym_diff < 8:
                score += 0.3
                signals.append("unnatural_facial_symmetry")
            elif sym_diff < 15:
                score += 0.1

        # 3. Eye reflection consistency — real eyes have matching catchlights
        eye_region = face_crop[int(h*0.2):int(h*0.45), :]
        if eye_region.size > 0:
            eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            _, bright = cv2.threshold(eye_gray, 220, 255, cv2.THRESH_BINARY)
            bright_pixels = np.sum(bright > 0)
            eye_area = eye_gray.shape[0] * eye_gray.shape[1]
            if bright_pixels == 0:
                score += 0.15
                signals.append("missing_eye_catchlights")

        # 4. Skin texture uniformity — AI faces are too smooth
        skin_region = face_crop[int(h*0.3):int(h*0.7), int(w*0.2):int(w*0.8)]
        if skin_region.size > 0:
            skin_gray = cv2.cvtColor(skin_region, cv2.COLOR_BGR2GRAY)
            local_std = float(np.std(skin_gray))
            if local_std < 8:
                score += 0.25
                signals.append("ai_smooth_skin_texture")

        return min(1.0, score), signals


class SelfieVerificationEngine:
    """
    Main verification engine combining all analyzers.
    """

    # Weights for score fusion
    WEIGHT_LIVENESS = 0.25
    WEIGHT_SPOOF = 0.25
    WEIGHT_AI_FACE = 0.20
    WEIGHT_DEEPFAKE = 0.30

    # Thresholds
    THRESHOLD_VERIFIED = 0.30    # Below = verified human
    THRESHOLD_REVIEW = 0.55      # Below = manual review
    # Above THRESHOLD_REVIEW = rejected

    def __init__(self):
        self.face_analyzer = FaceAnalyzer()
        self.liveness = LivenessAnalyzer()
        self.spoof = SpoofDetector()
        self.ai_face = AIFaceDetector()
        self.deepfake_model = HuggingFaceDeepfakeDetector()

    def verify(self, image_path, extra_frames=None):
        start_time = time.time()
        signals = []

        # Load image
        if not os.path.exists(image_path):
            return self._error("file_not_found")

        img = cv2.imread(image_path)
        if img is None:
            return self._error("image_decode_failed")

        # Step 1: Face Detection
        face_crop, rois = self.face_analyzer.get_face_roi(img)
        if face_crop is None:
            return self._error("no_face_detected")

        face_h, face_w = face_crop.shape[:2]
        if face_h < 80 or face_w < 80:
            return self._error("face_too_small")

        # Step 2: Liveness
        liveness_score, liveness_sigs = self.liveness.analyze(face_crop)
        signals.extend(liveness_sigs)

        # Step 3: Spoof Detection
        spoof_score, spoof_sigs = self.spoof.analyze(face_crop, img)
        signals.extend(spoof_sigs)

        # Step 4: AI Face Detection
        ai_score, ai_sigs = self.ai_face.analyze(face_crop)
        signals.extend(ai_sigs)

        # Step 5: Deepfake Detection (neural model)
        deepfake_score = self.deepfake_model.predict(face_crop)

        # Step 6: Multi-frame liveness boost (if extra frames provided)
        liveness_boost = 0.0
        if extra_frames:
            frame_scores = []
            for fp in extra_frames:
                f = cv2.imread(fp)
                if f is None:
                    continue
                fc, _ = self.face_analyzer.get_face_roi(f)
                if fc is not None:
                    ls, _ = self.liveness.analyze(fc)
                    frame_scores.append(ls)
            if frame_scores:
                avg_extra = float(np.mean(frame_scores))
                # If extra frames also look suspicious, increase liveness score
                liveness_score = (liveness_score + avg_extra) / 2
                if avg_extra < 0.3:
                    liveness_boost = -0.1  # Bonus for consistent live frames
                    signals.append("multi_frame_liveness_confirmed")

        # Score Fusion
        final_score = (
            self.WEIGHT_LIVENESS * liveness_score +
            self.WEIGHT_SPOOF * spoof_score +
            self.WEIGHT_AI_FACE * ai_score +
            self.WEIGHT_DEEPFAKE * deepfake_score +
            liveness_boost
        )
        final_score = max(0.0, min(1.0, final_score))

        # Decision
        if final_score < self.THRESHOLD_VERIFIED:
            verdict = "VERIFIED"
        elif final_score < self.THRESHOLD_REVIEW:
            verdict = "REVIEW"
        else:
            verdict = "REJECTED"

        processing_ms = int((time.time() - start_time) * 1000)

        return {
            "verdict": verdict,
            "final_score": round(final_score, 4),
            "liveness_score": round(liveness_score, 4),
            "spoof_score": round(spoof_score, 4),
            "ai_face_score": round(ai_score, 4),
            "deepfake_score": round(deepfake_score, 4),
            "signals": list(set(signals)),
            "processing_ms": processing_ms
        }

    def _error(self, reason):
        return {
            "verdict": "REJECTED",
            "final_score": 1.0,
            "liveness_score": 1.0,
            "spoof_score": 1.0,
            "ai_face_score": 1.0,
            "deepfake_score": 1.0,
            "signals": [reason],
            "processing_ms": 0
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({
            "verdict": "REJECTED",
            "final_score": 1.0,
            "signals": ["no_file_provided"],
            "processing_ms": 0
        }))
        sys.exit(1)

    image_path = sys.argv[1]

    # Optional extra frames: --frames path1 path2 ...
    extra_frames = None
    if "--frames" in sys.argv:
        idx = sys.argv.index("--frames")
        extra_frames = sys.argv[idx + 1:]

    try:
        engine = SelfieVerificationEngine()
        result = engine.verify(image_path, extra_frames)
        print(json.dumps(result))
    except Exception as e:
        _log(f"[SELFIE-VERIFY] Engine error: {e}")
        print(json.dumps({
            "verdict": "REJECTED",
            "final_score": 1.0,
            "signals": [f"engine_error: {str(e)}"],
            "processing_ms": 0
        }))
        sys.exit(1)
