import cv2
import numpy as np


class CompressionAnalyzer:
    def analyze(self, frame):
        """
        Detect JPEG/re-encoding artifacts that indicate manipulation.
        Deepfakes often show inconsistent compression patterns.
        """
        score = 0.0
        signals = []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. Block artifact detection (8x8 DCT block boundaries)
        block_diffs = []
        for y in range(7, h - 1, 8):
            row_diff = np.mean(np.abs(
                gray[y].astype(np.float32) - gray[y + 1].astype(np.float32)
            ))
            block_diffs.append(row_diff)

        if block_diffs:
            block_score = float(np.mean(block_diffs))
            if block_score > 15:
                score += 0.4
                signals.append("reencoding_block_artifacts")

        # 2. Double JPEG detection (histogram periodicity)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_normalized = hist / (hist.sum() + 1e-8)
        diffs = np.diff(hist_normalized)
        periodic_score = float(np.std(diffs))
        if periodic_score < 0.001:
            score += 0.3
            signals.append("double_compression_detected")

        return min(1.0, score), signals
