"""
Secondary Deepfake Review Engine
================================
Uses frequency-domain and GAN-artifact focused analysis — fundamentally
different from the primary pipeline (EfficientNet + HuggingFace face-level
ensemble) to catch deepfakes that the primary model missed.

Components:
  1. Frequency Spectrum Analysis  (weight 0.30)
  2. Cross-Patch Consistency      (weight 0.25)
  3. Noise Residual Analysis      (weight 0.20)
  4. Edge Coherence Analysis      (weight 0.15)
  5. EXIF Re-verification         (weight 0.10)

Invocation: python3 secondary_review.py <file_path>
Output: JSON to stdout (same interface as main.py)
"""

import sys
import os
import json
import numpy as np
import cv2

sys.path.append(os.path.dirname(__file__))

from config import THRESHOLD_APPROVE, THRESHOLD_REJECT
from core.detector import FaceAnalyzer

# Secondary review weights (different from primary)
WEIGHT_FREQUENCY = 0.30
WEIGHT_PATCH_CONSISTENCY = 0.25
WEIGHT_NOISE_RESIDUAL = 0.20
WEIGHT_EDGE_COHERENCE = 0.15
WEIGHT_EXIF_RECHECK = 0.10

# Patch config for cross-patch consistency
SECONDARY_PATCH_COUNT = 16
PATCH_SIZE = 64


class FrequencyAnalyzer:
    """Analyzes frequency spectrum for GAN artifacts."""

    def analyze(self, image: np.ndarray) -> tuple:
        signals = []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            gray = gray.astype(np.float32)

            # 2D FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            magnitude = np.log1p(magnitude)

            # Azimuthal averaging — radial power spectral density
            h, w = magnitude.shape
            cy, cx = h // 2, w // 2
            max_radius = min(cy, cx)

            radial_profile = []
            for r in range(1, max_radius):
                y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
                mask = (x ** 2 + y ** 2 >= (r - 1) ** 2) & (x ** 2 + y ** 2 < r ** 2)
                if mask.any():
                    radial_profile.append(np.mean(magnitude[mask]))

            if len(radial_profile) < 10:
                return 0.0, signals

            radial_profile = np.array(radial_profile)

            # Natural images follow ~1/f power law (linear in log-log)
            # GAN images deviate — check slope consistency
            log_freq = np.log(np.arange(1, len(radial_profile) + 1))
            log_power = np.log(radial_profile + 1e-10)

            # Linear regression in log-log space
            coeffs = np.polyfit(log_freq, log_power, 1)
            slope = coeffs[0]
            residuals = log_power - np.polyval(coeffs, log_freq)
            residual_std = np.std(residuals)

            # High-frequency anomaly check
            mid = len(radial_profile) // 2
            high_freq_ratio = np.mean(radial_profile[mid:]) / (np.mean(radial_profile[:mid]) + 1e-10)

            # Score computation
            score = 0.0

            # Natural images have slope ~ -1.0 to -2.0. GANs often have flatter slopes
            if slope > -0.5:
                score += 0.4
                signals.append("flat_frequency_slope")
            elif slope > -0.8:
                score += 0.2

            # High residual variance suggests non-natural frequency distribution
            if residual_std > 1.0:
                score += 0.3
                signals.append("irregular_frequency_distribution")
            elif residual_std > 0.5:
                score += 0.15

            # Unusual high-frequency energy ratio
            if high_freq_ratio > 0.3:
                score += 0.3
                signals.append("elevated_high_frequency_energy")
            elif high_freq_ratio > 0.15:
                score += 0.1

            return min(1.0, score), signals

        except Exception:
            return 0.0, signals


class PatchConsistencyAnalyzer:
    """Measures cross-patch score variance to detect localized manipulation."""

    def __init__(self):
        self.face_analyzer = FaceAnalyzer()

    def analyze(self, image: np.ndarray, face_crop: np.ndarray) -> tuple:
        signals = []
        try:
            h, w = face_crop.shape[:2]
            if h < PATCH_SIZE or w < PATCH_SIZE:
                return 0.0, signals

            patches = []
            step_y = max(1, (h - PATCH_SIZE) // (int(SECONDARY_PATCH_COUNT ** 0.5) - 1))
            step_x = max(1, (w - PATCH_SIZE) // (int(SECONDARY_PATCH_COUNT ** 0.5) - 1))

            for y in range(0, h - PATCH_SIZE + 1, step_y):
                for x in range(0, w - PATCH_SIZE + 1, step_x):
                    patch = face_crop[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                    patches.append(patch)
                    if len(patches) >= SECONDARY_PATCH_COUNT:
                        break
                if len(patches) >= SECONDARY_PATCH_COUNT:
                    break

            if len(patches) < 4:
                return 0.0, signals

            # Compute per-patch statistics
            patch_means = [np.mean(p) for p in patches]
            patch_stds = [np.std(p) for p in patches]

            # Compute local noise level per patch using Laplacian
            patch_noise = []
            for p in patches:
                gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY) if len(p.shape) == 3 else p
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                patch_noise.append(laplacian_var)

            noise_std = np.std(patch_noise)
            noise_mean = np.mean(patch_noise)
            noise_cv = noise_std / (noise_mean + 1e-10)  # Coefficient of variation

            # GAN-generated faces have very uniform noise across patches
            # Real faces or face-swaps show more local variation
            score = 0.0

            # Very low noise variance → suspiciously uniform (GAN-like)
            if noise_cv < 0.15:
                score += 0.5
                signals.append("uniform_patch_noise_gan_signature")
            elif noise_cv < 0.3:
                score += 0.2

            # Check for abrupt transitions between patches (face-swap boundary)
            mean_variance = np.std(patch_means)
            if mean_variance > 40:
                score += 0.3
                signals.append("patch_intensity_discontinuity")
            elif mean_variance > 25:
                score += 0.15

            # Check texture consistency
            std_variance = np.std(patch_stds)
            if std_variance > 20:
                score += 0.2
                signals.append("inconsistent_patch_texture")

            return min(1.0, score), signals

        except Exception:
            return 0.0, signals


class NoiseResidualAnalyzer:
    """SRM-style noise residual analysis for detecting manipulation artifacts."""

    def analyze(self, image: np.ndarray) -> tuple:
        signals = []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            gray = gray.astype(np.float64)

            # SRM-style high-pass filter kernels
            kernel_1 = np.array([[-1, 2, -1],
                                 [2, -4, 2],
                                 [-1, 2, -1]], dtype=np.float64)

            kernel_2 = np.array([[0, 0, -1, 0, 0],
                                 [0, 0, 2, 0, 0],
                                 [-1, 2, -4, 2, -1],
                                 [0, 0, 2, 0, 0],
                                 [0, 0, -1, 0, 0]], dtype=np.float64)

            # Extract noise residuals
            residual_1 = cv2.filter2D(gray, -1, kernel_1)
            residual_2 = cv2.filter2D(gray, -1, kernel_2)

            # Analyze noise residual statistics
            r1_std = np.std(residual_1)
            r2_std = np.std(residual_2)

            # Co-occurrence analysis on quantized residual
            quantized = np.clip(np.round(residual_1 / (r1_std + 1e-10) * 2), -4, 4).astype(np.int8)

            # Horizontal co-occurrence
            h, w = quantized.shape
            if h > 1 and w > 1:
                left = quantized[:, :-1].flatten()
                right = quantized[:, 1:].flatten()

                # Build co-occurrence matrix (9x9 for values -4 to 4)
                co_matrix = np.zeros((9, 9), dtype=np.float64)
                for l, r in zip(left[:10000], right[:10000]):  # Sample for speed
                    co_matrix[l + 4, r + 4] += 1

                co_matrix /= (co_matrix.sum() + 1e-10)

                # Entropy of co-occurrence
                co_flat = co_matrix.flatten()
                co_flat = co_flat[co_flat > 0]
                entropy = -np.sum(co_flat * np.log2(co_flat + 1e-10))

                # Symmetry check
                symmetry = np.sum(np.abs(co_matrix - co_matrix.T)) / 2

            else:
                entropy = 0
                symmetry = 0

            score = 0.0

            # Low noise residual std → content may be synthetically generated
            if r1_std < 2.0:
                score += 0.3
                signals.append("low_noise_residual_synthetic")
            elif r1_std < 5.0:
                score += 0.1

            # Very high residual → heavy post-processing or manipulation
            if r1_std > 30.0:
                score += 0.2
                signals.append("excessive_noise_residual")

            # Low entropy → uniform noise pattern (GAN artifact)
            if entropy < 2.5:
                score += 0.3
                signals.append("low_noise_entropy_gan")
            elif entropy < 3.5:
                score += 0.1

            # Asymmetric co-occurrence → directional artifacts
            if symmetry > 0.1:
                score += 0.2
                signals.append("asymmetric_noise_pattern")

            return min(1.0, score), signals

        except Exception:
            return 0.0, signals


class EdgeCoherenceAnalyzer:
    """Checks boundary consistency around face region for face-swap detection."""

    def analyze(self, image: np.ndarray, face_crop: np.ndarray) -> tuple:
        signals = []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop

            # Multi-threshold Canny edge detection
            edges_low = cv2.Canny(gray, 30, 80)
            edges_mid = cv2.Canny(gray, 50, 150)
            edges_high = cv2.Canny(gray, 100, 200)

            # Face interior edge density
            face_edges = cv2.Canny(face_gray, 50, 150)
            face_edge_density = np.sum(face_edges > 0) / (face_edges.size + 1e-10)

            # Create face boundary mask (dilate - erode to get border)
            h, w = face_gray.shape
            mask = np.zeros_like(gray)
            # Estimate face position (center of image for face crop analysis)
            cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
            fh, fw = h // 2, w // 2
            y1, y2 = max(0, cy - fh), min(gray.shape[0], cy + fh)
            x1, x2 = max(0, cx - fw), min(gray.shape[1], cx + fw)
            mask[y1:y2, x1:x2] = 255

            # Boundary region (dilated - original)
            kernel = np.ones((15, 15), np.uint8)
            dilated = cv2.dilate(mask, kernel, iterations=1)
            boundary = dilated - mask

            # Edge density at boundary vs interior
            boundary_pixels = boundary > 0
            if np.sum(boundary_pixels) > 0:
                boundary_edge_density = np.sum(edges_mid[boundary_pixels] > 0) / np.sum(boundary_pixels)
            else:
                boundary_edge_density = 0

            score = 0.0

            # High boundary edge density relative to face interior = potential splicing
            edge_ratio = boundary_edge_density / (face_edge_density + 1e-10)
            if edge_ratio > 3.0:
                score += 0.5
                signals.append("boundary_edge_discontinuity")
            elif edge_ratio > 1.5:
                score += 0.2

            # Very low face edge density = possibly AI-generated smooth face
            if face_edge_density < 0.02:
                score += 0.3
                signals.append("smooth_face_low_edges")
            elif face_edge_density < 0.05:
                score += 0.1

            # Edge consistency across thresholds
            low_count = np.sum(edges_low > 0)
            high_count = np.sum(edges_high > 0)
            if low_count > 0:
                threshold_ratio = high_count / low_count
                if threshold_ratio < 0.1:
                    score += 0.2
                    signals.append("edge_threshold_instability")

            return min(1.0, score), signals

        except Exception:
            return 0.0, signals


class EXIFReverifier:
    """Deeper EXIF/container analysis beyond the primary metadata scan."""

    def analyze(self, file_path: str) -> tuple:
        signals = []
        try:
            score = 0.0

            # Read raw bytes for JPEG analysis
            with open(file_path, 'rb') as f:
                header = f.read(32)
                f.seek(0)
                raw = f.read()

            is_jpeg = header[:2] == b'\xff\xd8'

            if is_jpeg:
                # Check for double JPEG compression via quantization tables
                dqt_count = raw.count(b'\xff\xdb')
                if dqt_count > 2:
                    score += 0.4
                    signals.append("multiple_quantization_tables")

                # Check for thumbnail inconsistency
                # JFIF thumbnail marker
                jfif_pos = raw.find(b'JFIF')
                exif_pos = raw.find(b'Exif')

                if jfif_pos >= 0 and exif_pos >= 0:
                    # Both JFIF and EXIF present — unusual for camera originals
                    score += 0.2
                    signals.append("mixed_jfif_exif_headers")

                # Check for editing software signatures
                software_markers = [b'Photoshop', b'GIMP', b'Adobe', b'FaceApp',
                                    b'Remini', b'deepfake', b'faceswap']
                for marker in software_markers:
                    if marker.lower() in raw[:4096].lower():
                        score += 0.4
                        signals.append(f"editing_software_detected_{marker.decode()}")
                        break

            else:
                # PNG or other format — check for suspicious chunks
                is_png = header[:4] == b'\x89PNG'
                if is_png:
                    # Check for tEXt chunks with editing software
                    text_pos = raw.find(b'tEXt')
                    if text_pos >= 0:
                        text_chunk = raw[text_pos:text_pos + 256]
                        software_markers = [b'Photoshop', b'GIMP', b'Adobe']
                        for marker in software_markers:
                            if marker in text_chunk:
                                score += 0.3
                                signals.append("png_editing_software_detected")
                                break

            # File size anomaly check
            file_size = len(raw)
            img = cv2.imread(file_path)
            if img is not None:
                pixel_count = img.shape[0] * img.shape[1]
                bytes_per_pixel = file_size / (pixel_count + 1e-10)

                # Extremely low compression = re-saved at max quality (common for deepfakes)
                if is_jpeg and bytes_per_pixel > 3.0:
                    score += 0.2
                    signals.append("unusually_low_compression")

            return min(1.0, score), signals

        except Exception:
            return 0.0, signals


class SecondaryReviewEngine:
    """Secondary deepfake detection using frequency-domain and GAN-artifact analysis."""

    def __init__(self):
        self.frequency_analyzer = FrequencyAnalyzer()
        self.patch_analyzer = PatchConsistencyAnalyzer()
        self.noise_analyzer = NoiseResidualAnalyzer()
        self.edge_analyzer = EdgeCoherenceAnalyzer()
        self.exif_analyzer = EXIFReverifier()
        self.face_analyzer = FaceAnalyzer()

    def run(self, file_path: str) -> dict:
        signals_list = []

        # Load image
        image = cv2.imread(file_path)
        if image is None:
            return self._finalize(0.0, 0.0, 0.0, 0.0, 0.0, 0.1,
                                  ["media_decode_error"])

        # Detect face
        face_crop, rois = self.face_analyzer.get_face_roi(image)
        if face_crop is None:
            # No face — analyze full image
            face_crop = image
            signals_list.append("no_face_using_full_image")

        # 1. Frequency Spectrum Analysis
        freq_score, freq_sigs = self.frequency_analyzer.analyze(face_crop)
        signals_list.extend(freq_sigs)

        # 2. Cross-Patch Consistency
        patch_score, patch_sigs = self.patch_analyzer.analyze(image, face_crop)
        signals_list.extend(patch_sigs)

        # 3. Noise Residual Analysis
        noise_score, noise_sigs = self.noise_analyzer.analyze(face_crop)
        signals_list.extend(noise_sigs)

        # 4. Edge Coherence
        edge_score, edge_sigs = self.edge_analyzer.analyze(image, face_crop)
        signals_list.extend(edge_sigs)

        # 5. EXIF Re-verification
        exif_score, exif_sigs = self.exif_analyzer.analyze(file_path)
        signals_list.extend(exif_sigs)

        # Weighted fusion
        secondary_score = (
            (WEIGHT_FREQUENCY * freq_score) +
            (WEIGHT_PATCH_CONSISTENCY * patch_score) +
            (WEIGHT_NOISE_RESIDUAL * noise_score) +
            (WEIGHT_EDGE_COHERENCE * edge_score) +
            (WEIGHT_EXIF_RECHECK * exif_score)
        )

        return self._finalize(
            freq_score, patch_score, noise_score,
            edge_score, exif_score, secondary_score, signals_list
        )

    def _finalize(self, frequency_score, gan_artifact_score, noise_consistency_score,
                  edge_coherence_score, exif_score, secondary_score, signals):
        """
        Decision thresholds (same 3-tier as primary):
          secondary_score >= 0.80 -> REMOVE (confirmed deepfake)
          0.60 <= score < 0.80    -> MANUAL_REVIEW
          score < 0.60            -> RESTORE (content is real)
        """
        verdict = "APPROVED"
        if secondary_score >= THRESHOLD_REJECT:
            verdict = "REJECTED"
            signals.append("secondary_confirmed_deepfake")
        elif secondary_score >= THRESHOLD_APPROVE:
            verdict = "UNDER_REVIEW"
            signals.append("secondary_borderline_needs_review")

        unique_signals = list(set(signals))

        return {
            "model": "secondary-frequency-gan-ensemble",
            "frequency_score": round(frequency_score, 4),
            "gan_artifact_score": round(gan_artifact_score, 4),
            "noise_consistency_score": round(noise_consistency_score, 4),
            "edge_coherence_score": round(edge_coherence_score, 4),
            "patch_variance_score": round(exif_score, 4),
            "secondary_score": round(secondary_score, 4),
            "verdict": verdict,
            "signals": unique_signals
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({
            "model": "secondary-frequency-gan-ensemble",
            "frequency_score": 0.0,
            "gan_artifact_score": 0.0,
            "noise_consistency_score": 0.0,
            "edge_coherence_score": 0.0,
            "patch_variance_score": 0.0,
            "secondary_score": 0.1,
            "verdict": "REJECTED",
            "signals": ["no_file_provided"]
        }))
        sys.exit(1)

    try:
        engine = SecondaryReviewEngine()
        report = engine.run(sys.argv[1])
        print(json.dumps(report))
    except Exception as e:
        print(json.dumps({
            "model": "secondary-frequency-gan-ensemble",
            "frequency_score": 0.0,
            "gan_artifact_score": 0.0,
            "noise_consistency_score": 0.0,
            "edge_coherence_score": 0.0,
            "patch_variance_score": 0.0,
            "secondary_score": 0.1,
            "verdict": "REJECTED",
            "signals": [f"engine_error: {str(e)}", "fail_closed"]
        }))
        sys.exit(1)
