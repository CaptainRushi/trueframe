import sys
import os
import json
import numpy as np
import cv2

# Ensure we can import from core/
sys.path.append(os.path.dirname(__file__))

from config import *
from core.precheck import MetadataScanner
from core.extractor import FrameExtractor
from core.detector import FaceAnalyzer
from core.models import EfficientNetONNXDetector
from core.heuristics import ArtifactAnalyzer
from core.temporal import TemporalAnalyzer

class DeepfakeGuardProcess:
    def __init__(self):
        self.metadata_scanner = MetadataScanner()
        self.frame_extractor = FrameExtractor()
        self.face_analyzer = FaceAnalyzer()
        self.model = EfficientNetONNXDetector(MODEL_PATH)
        self.artifact_analyzer = ArtifactAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()

    def run(self, file_path):
        """
        Main verification flow as per Master Prompt.
        """
        signals_list = []
        
        # 1. Metadata Scan (WEIGHT_METADATA = 0.10)
        meta_score, meta_sigs = self.metadata_scanner.scan(file_path)
        if meta_score >= 0.8:
            signals_list.append("suspicious_metadata_integrity")

        # 2. Extract Frames (Max 32 frames @ 1 FPS)
        frames, timestamps = self.frame_extractor.extract_frames(file_path)
        is_video = len(frames) > 1
        
        if not frames:
             # Try as image
             img = cv2.imread(file_path)
             if img is not None:
                 frames = [img]
                 timestamps = [0.0]
             else:
                return self._finalize_verdict(1.0, 1.0, ["media_decode_error"])

        # 3. Process Frames & Detect Faces
        valid_crops = []
        frame_artifact_scores = []
        
        for frame in frames:
            face_crop, rois = self.face_analyzer.get_face_roi(frame)
            if face_crop is not None:
                valid_crops.append(face_crop)
                # Artifact score for this frame
                art_score, art_sigs = self.artifact_analyzer.analyze(rois)
                frame_artifact_scores.append(art_score)
                if art_score > 0.7:
                    signals_list.append("high_frequency_artifacts")

        if not valid_crops:
            # FAIL-CLOSED: No faces found
            return self._finalize_verdict(1.0, 1.0, ["no_clear_faces_detected"])

        # 4. EfficientNet Inference (Batch)
        model_scores = self.model.predict_batch(valid_crops)
        avg_model_score = float(np.mean(model_scores))
        
        # 5. Heuristics & Temporal Analysis
        avg_artifact_score = float(np.mean(frame_artifact_scores))
        
        temporal_risk = 0.0
        if is_video:
            # Compute variance and instability for temporal risk
            # We wrap frame results for TemporalAnalyzer
            mock_frame_results = [{"hf_score": s, "artifact_score": a} for s, a in zip(model_scores, frame_artifact_scores)]
            temporal_risk, temp_sigs = self.temporal_analyzer.analyze(mock_frame_results)
            if temporal_risk > 0.4:
                signals_list.append("temporal_face_distortion")
        
        # 6. SCORE FUSION (MANDATORY)
        # final_score = (0.45 * model_score) + (0.25 * artifact_score) + (0.20 * temporal_score) + (0.10 * metadata_score)
        final_score = (
            (WEIGHT_MODEL * avg_model_score) +
            (WEIGHT_ARTIFACT * avg_artifact_score) +
            (WEIGHT_TEMPORAL * temporal_risk) +
            (WEIGHT_METADATA * meta_score)
        )

        return self._finalize_verdict(avg_model_score, final_score, signals_list)

    def _finalize_verdict(self, model_score, final_score, signals):
        """
        Decision Thresholds:
        final_score >= 0.60 -> REJECT (Deepfake)
        0.40 – 0.59        -> REJECT (Unsafe)
        < 0.40             -> APPROVE
        """
        verdict = "APPROVED"
        if final_score >= THRESHOLD_REJECT:
            verdict = "REJECTED"
            # Add specific signal if rejected as deepfake
            if "deepfake_detected" not in signals:
                signals.append("synthetic_generation_signal")
        elif final_score >= THRESHOLD_UNSAFE:
            verdict = "REJECTED"
            if "unsafe_content" not in signals:
                signals.append("high_manipulation_risk")

        # Deduplicate signals
        unique_signals = list(set(signals))

        report = {
            "model": "efficientnet-b0",
            "model_score": round(model_score, 4),
            "final_score": round(final_score, 4),
            "verdict": verdict,
            "signals": unique_signals
        }
        return report

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({
            "model": "efficientnet-b0",
            "model_score": 1.0,
            "final_score": 1.0,
            "verdict": "REJECTED",
            "signals": ["no_file_provided"]
        }))
        sys.exit(1)

    try:
        guard = DeepfakeGuardProcess()
        report = guard.run(sys.argv[1])
        print(json.dumps(report))
    except Exception as e:
        # FAIL-CLOSED
        print(json.dumps({
            "model": "efficientnet-b0",
            "model_score": 1.0,
            "final_score": 1.0,
            "verdict": "REJECTED",
            "signals": [f"engine_error: {str(e)}", "fail_closed"]
        }))
        sys.exit(1)
