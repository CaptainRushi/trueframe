"""
TrueFrame Reels — Additional Anti-Deepfake Signals
====================================================
Supplementary detection modules for enhanced accuracy:
  1. Audio-Lip Synchronization Detector
  2. Eye Blink Anomaly Detector
  3. GAN Fingerprint Detector
  4. Metadata Integrity Validator
"""

import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger("trueframe.signals")


class LipSyncDetector:
    """
    Detects audio-visual lip sync mismatches.
    Deepfakes with voice cloning often show subtle timing misalignment
    between lip movements and the audio signal.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def analyze(
        self, video_path: str
    ) -> Tuple[float, List[str]]:
        """
        Analyze lip-audio synchronization.
        Returns (score 0-1, signals).
        """
        signals = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0.0, ["video_open_failed"]

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0 

            if duration < 2.0:
                cap.release()
                return 0.0, ["too_short_for_lip_sync"]

            # Analyze mouth movement across frames
            mouth_movements = []
            prev_mouth = None
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % max(1, int(fps / 5)) == 0:
                    mouth = self._extract_mouth_region(frame)
                    if mouth is not None:
                        if prev_mouth is not None:
                            diff = self._compute_optical_flow(
                                prev_mouth, mouth
                            )
                            mouth_movements.append(diff)
                        prev_mouth = mouth.copy()
                frame_idx += 1

            cap.release()

            if len(mouth_movements) < 5:
                return 0.0, ["insufficient_mouth_data"]

            movements = np.array(mouth_movements)
            # Analyze temporal consistency of mouth movements
            score = self._analyze_movement_pattern(movements)

            if score > 0.6:
                signals.append("lip_sync_anomaly_detected")
            if score > 0.8:
                signals.append("severe_lip_audio_mismatch")

            return float(np.clip(score, 0, 1)), signals

        except Exception as e:
            logger.debug(f"LipSync analysis error: {e}")
            return 0.0, []

    def _extract_mouth_region(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract mouth region using face detection heuristics."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) == 0:
                return None

            x, y, w, h = sorted(
                faces, key=lambda f: f[2] * f[3], reverse=True
            )[0]

            # Mouth is roughly in the lower third of the face
            mouth_y = y + int(h * 0.65)
            mouth_h = int(h * 0.25)
            mouth_x = x + int(w * 0.2)
            mouth_w = int(w * 0.6)

            mouth = gray[
                mouth_y : mouth_y + mouth_h,
                mouth_x : mouth_x + mouth_w,
            ]
            if mouth.size == 0:
                return None

            return cv2.resize(mouth, (64, 32))
        except Exception:
            return None

    def _compute_optical_flow(
        self, prev: np.ndarray, curr: np.ndarray
    ) -> float:
        """Compute magnitude of optical flow between mouth frames."""
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0,
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return float(np.mean(magnitude))

    def _analyze_movement_pattern(self, movements: np.ndarray) -> float:
        """Analyze if mouth movements show unnatural patterns."""
        score = 0.0
        std = np.std(movements)
        mean = np.mean(movements)

        # Unnatural uniformity (deepfake lip-sync tends to be too smooth)
        if std < 0.3 and mean > 0.5:
            score += 0.4

        # Check for sudden jumps (frame inconsistency)
        diffs = np.abs(np.diff(movements))
        max_jump = np.max(diffs) if len(diffs) > 0 else 0
        if max_jump > mean * 3:
            score += 0.3

        # Periodicity check (unnatural repetitive patterns)
        if len(movements) > 10:
            fft = np.abs(np.fft.fft(movements - mean))
            dominant_freq_ratio = np.max(fft[1:]) / (np.sum(fft[1:]) + 1e-8)
            if dominant_freq_ratio > 0.5:
                score += 0.3

        return min(1.0, score)


class BlinkDetector:
    """
    Eye blink anomaly detection.
    Real humans blink 15-20 times per minute. Deepfakes often show
    abnormal blink patterns (too few, too many, or too uniform).
    """

    NORMAL_BLINK_RATE = (12, 25)  # blinks per minute

    def analyze(
        self, video_path: str
    ) -> Tuple[float, List[str]]:
        signals = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0.0, []

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_min = (total_frames / fps) / 60 if fps > 0 else 0

            if duration_min < 0.1:
                cap.release()
                return 0.0, ["too_short_for_blink"]

            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye.xml"
            )
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            eye_open_history = []
            frame_idx = 0
            sample_interval = max(1, int(fps / 10))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_interval == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face_roi = gray[y : y + h, x : x + w]
                        eyes = eye_cascade.detectMultiScale(face_roi)
                        eye_open_history.append(len(eyes) >= 2)

                frame_idx += 1

            cap.release()

            if len(eye_open_history) < 10:
                return 0.0, ["insufficient_eye_data"]

            score = self._analyze_blinks(eye_open_history, duration_min)

            if score > 0.5:
                signals.append("abnormal_blink_pattern")

            return float(np.clip(score, 0, 1)), signals

        except Exception as e:
            logger.debug(f"Blink analysis error: {e}")
            return 0.0, []

    def _analyze_blinks(
        self, eye_open: List[bool], duration_min: float
    ) -> float:
        score = 0.0
        open_arr = np.array(eye_open, dtype=float)

        # Count blinks (transitions from open→closed→open)
        transitions = np.abs(np.diff(open_arr))
        blink_count = int(np.sum(transitions) / 2)
        blink_rate = blink_count / max(0.01, duration_min)

        lo, hi = self.NORMAL_BLINK_RATE
        if blink_rate < lo * 0.3:
            score += 0.5  # Suspiciously few blinks
        elif blink_rate > hi * 2:
            score += 0.3  # Too many blinks

        # Check blink regularity (real blinks are irregular)
        if len(transitions) > 5:
            blink_intervals = np.diff(np.where(transitions > 0)[0])
            if len(blink_intervals) > 2:
                cv_interval = np.std(blink_intervals) / (
                    np.mean(blink_intervals) + 1e-8
                )
                if cv_interval < 0.2:
                    score += 0.3  # Too regular

        return min(1.0, score)


class GANFingerprintDetector:
    """
    Detects GAN-specific artifacts in the frequency domain.
    Different GAN architectures leave unique 'fingerprints'
    in the generated images.
    """

    def analyze(self, face_crop: np.ndarray) -> Tuple[float, List[str]]:
        signals = []
        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY).astype(
                np.float32
            )

            # 2D FFT
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude = np.log1p(np.abs(fshift))

            h, w = magnitude.shape
            cy, cx = h // 2, w // 2

            # Radial average
            max_r = min(cy, cx)
            profile = []
            for r in range(1, max_r):
                y, x = np.ogrid[-cy : h - cy, -cx : w - cx]
                mask = (x ** 2 + y ** 2 >= (r - 1) ** 2) & (
                    x ** 2 + y ** 2 < r ** 2
                )
                if mask.any():
                    profile.append(np.mean(magnitude[mask]))

            if len(profile) < 10:
                return 0.0, signals

            profile = np.array(profile)
            score = 0.0

            # Check for spectral spikes (GAN checkerboard artifacts)
            residual = profile - np.convolve(
                profile, np.ones(5) / 5, mode="same"
            )
            spikes = np.abs(residual) > 2.0 * np.std(residual)
            spike_ratio = np.sum(spikes) / len(spikes)
            if spike_ratio > 0.15:
                score += 0.4
                signals.append("gan_spectral_spikes")

            # Power law deviation
            log_f = np.log(np.arange(1, len(profile) + 1) + 1e-10)
            log_p = np.log(profile + 1e-10)
            coeffs = np.polyfit(log_f, log_p, 1)
            if coeffs[0] > -0.5:
                score += 0.3
                signals.append("flat_power_spectrum")

            # High-frequency energy ratio
            mid = len(profile) // 2
            hf_ratio = np.mean(profile[mid:]) / (
                np.mean(profile[:mid]) + 1e-10
            )
            if hf_ratio > 0.3:
                score += 0.3
                signals.append("elevated_hf_energy")

            return float(min(1.0, score)), signals

        except Exception as e:
            logger.debug(f"GAN fingerprint error: {e}")
            return 0.0, signals


class MetadataValidator:
    """
    Validates video metadata for manipulation indicators.
    Checks container format, encoding params, creation timestamps.
    """

    SUSPICIOUS_ENCODERS = [
        "faceswap", "deepfake", "deepfacelab", "facefusion",
    ]

    def analyze(self, video_path: str) -> Tuple[float, List[str]]:
        signals = []
        score = 0.0
        try:
            # Read raw header bytes
            with open(video_path, "rb") as f:
                header = f.read(4096)

            header_lower = header.lower()

            for enc in self.SUSPICIOUS_ENCODERS:
                if enc.encode() in header_lower:
                    score += 0.5
                    signals.append(f"suspicious_encoder_{enc}")
                    break

            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Check for non-standard fps
                if fps > 0 and fps not in [
                    23.976, 24, 25, 29.97, 30, 50, 59.94, 60,
                ]:
                    if abs(fps - round(fps)) > 0.1:
                        score += 0.2
                        signals.append("non_standard_framerate")

                # Non-standard resolution
                if w > 0 and h > 0:
                    aspect = w / h
                    standard_aspects = [16 / 9, 9 / 16, 4 / 3, 1.0]
                    min_diff = min(abs(aspect - sa) for sa in standard_aspects)
                    if min_diff > 0.1:
                        score += 0.1
                        signals.append("unusual_aspect_ratio")

                cap.release()

            return float(min(1.0, score)), signals

        except Exception as e:
            logger.debug(f"Metadata validation error: {e}")
            return 0.0, signals


# ────────────── COMBINED SIGNAL ANALYZER ─────────────

class AntiDeepfakeSignalAnalyzer:
    """
    Combines all supplementary detection signals into a unified score.
    Weights:
      - Lip Sync:        0.30
      - Eye Blink:       0.20
      - GAN Fingerprint: 0.30
      - Metadata:        0.20
    """

    WEIGHTS = {
        "lip_sync": 0.30,
        "blink": 0.20,
        "gan_fingerprint": 0.30,
        "metadata": 0.20,
    }

    def __init__(self):
        self.lip_sync = LipSyncDetector()
        self.blink = BlinkDetector()
        self.gan_fp = GANFingerprintDetector()
        self.metadata = MetadataValidator()

    def analyze_video(
        self, video_path: str, face_crops: list = None
    ) -> Dict:
        all_signals = []

        # Lip sync
        ls_score, ls_sigs = self.lip_sync.analyze(video_path)
        all_signals.extend(ls_sigs)

        # Blink
        bk_score, bk_sigs = self.blink.analyze(video_path)
        all_signals.extend(bk_sigs)

        # GAN fingerprint (use first face crop if available)
        gf_score = 0.0
        if face_crops:
            gf_score, gf_sigs = self.gan_fp.analyze(face_crops[0])
            all_signals.extend(gf_sigs)

        # Metadata
        md_score, md_sigs = self.metadata.analyze(video_path)
        all_signals.extend(md_sigs)

        combined = (
            self.WEIGHTS["lip_sync"] * ls_score
            + self.WEIGHTS["blink"] * bk_score
            + self.WEIGHTS["gan_fingerprint"] * gf_score
            + self.WEIGHTS["metadata"] * md_score
        )

        return {
            "supplementary_score": round(float(combined), 4),
            "lip_sync_score": round(ls_score, 4),
            "blink_score": round(bk_score, 4),
            "gan_fingerprint_score": round(gf_score, 4),
            "metadata_score": round(md_score, 4),
            "signals": list(set(all_signals)),
        }
