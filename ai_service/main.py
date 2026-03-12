import sys
import os
import json
import numpy as np
import cv2

sys.path.append(os.path.dirname(__file__))

from config import *
from core.precheck import MetadataScanner
from core.extractor import FrameExtractor
from core.detector import FaceAnalyzer
from core.models import EfficientNetONNXDetector, HuggingFaceDeepfakeDetector
from core.heuristics import ArtifactAnalyzer
from core.temporal import TemporalAnalyzer
from core.compression import CompressionAnalyzer
from core.patches import PatchVoter


class DeepfakeGuardProcess:
    def __init__(self):
        self.metadata_scanner = MetadataScanner()
        self.frame_extractor = FrameExtractor()
        self.face_analyzer = FaceAnalyzer()
        self.model = EfficientNetONNXDetector(MODEL_PATH)
        self.hf_model = HuggingFaceDeepfakeDetector()
        self.artifact_analyzer = ArtifactAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()
        self.compression_analyzer = CompressionAnalyzer()
        self.patch_voter = PatchVoter()

    def run(self, file_path):
        """
        Main verification flow.
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
            img = cv2.imread(file_path)
            if img is not None:
                frames = [img]
                timestamps = [0.0]
            else:
                return self._finalize_verdict(0.1, 0.0, 0.0, meta_score, 0.0, 0.1, ["media_decode_error"])

        # 3. Process Frames & Detect Faces
        valid_crops = []
        frame_artifact_scores = []
        frame_compression_scores = []

        for frame in frames:
            face_crop, rois = self.face_analyzer.get_face_roi(frame)
            if face_crop is not None:
                valid_crops.append(face_crop)
                # Artifact score for this frame
                art_score, art_sigs = self.artifact_analyzer.analyze(rois)
                frame_artifact_scores.append(art_score)
                if art_score > 0.7:
                    signals_list.append("high_frequency_artifacts")
                # Compression artifact score for this frame
                comp_score, comp_sigs = self.compression_analyzer.analyze(face_crop)
                frame_compression_scores.append(comp_score)
                signals_list.extend(comp_sigs)

        if not valid_crops:
            return self._finalize_verdict(0.1, 0.0, 0.0, meta_score, 0.0, 0.1, ["no_clear_faces_detected"])

        # 4. Neural Models Inference (Ensemble)
        # For single images, use multi-patch voting to reduce false positives
        if not is_video and len(valid_crops) == 1:
            patches = self.patch_voter.extract_patches(valid_crops[0])
            all_crops = valid_crops + patches
        else:
            all_crops = valid_crops

        onnx_scores = self.model.predict_batch(all_crops)
        hf_scores = self.hf_model.predict_batch(all_crops)

        # Average fusion for higher security
        model_scores = [(o + h) / 2 for o, h in zip(onnx_scores, hf_scores)]
        avg_model_score = float(np.mean(model_scores))

        # 5. Heuristics & Temporal Analysis
        avg_artifact_score = float(np.mean(frame_artifact_scores))
        avg_compression_score = float(np.mean(frame_compression_scores)) if frame_compression_scores else 0.0

        temporal_risk = 0.0
        if is_video:
            mock_frame_results = [{"hf_score": s, "artifact_score": a} for s, a in zip(model_scores, frame_artifact_scores)]
            temporal_risk, temp_sigs = self.temporal_analyzer.analyze(mock_frame_results)
            if temporal_risk > 0.4:
                signals_list.append("temporal_face_distortion")

        # 6. SCORE FUSION
        final_score = (
            (WEIGHT_MODEL * avg_model_score) +
            (WEIGHT_ARTIFACT * avg_artifact_score) +
            (WEIGHT_TEMPORAL * temporal_risk) +
            (WEIGHT_METADATA * meta_score) +
            (WEIGHT_COMPRESSION * avg_compression_score)
        )

        return self._finalize_verdict(
            avg_model_score, avg_artifact_score, temporal_risk,
            meta_score, avg_compression_score, final_score, signals_list
        )

    def _finalize_verdict(self, model_score, artifact_score, temporal_score,
                          metadata_score, compression_score, final_score, signals):
        """
        Decision Thresholds (3-tier):
        final_score >= 0.80 -> REJECTED (Deepfake)
        0.60 – 0.79        -> UNDER_REVIEW (needs manual review)
        < 0.60             -> APPROVED (real)
        """
        verdict = "APPROVED"
        if final_score >= THRESHOLD_REJECT:
            verdict = "REJECTED"
            if "deepfake_detected" not in signals:
                signals.append("synthetic_generation_signal")
        elif final_score >= THRESHOLD_APPROVE:
            verdict = "UNDER_REVIEW"
            if "needs_review" not in signals:
                signals.append("borderline_score_needs_review")

        unique_signals = list(set(signals))

        report = {
            "model": "efficientnet+huggingface-ensemble",
            "model_score": round(model_score, 4),
            "artifact_score": round(artifact_score, 4),
            "temporal_score": round(temporal_score, 4),
            "metadata_score": round(metadata_score, 4),
            "compression_score": round(compression_score, 4),
            "final_score": round(final_score, 4),
            "verdict": verdict,
            "signals": unique_signals
        }
        return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # No file provided — still output valid JSON so backend can parse it
        print(json.dumps({
            "model": "efficientnet+huggingface-ensemble",
            "model_score": 0.1,
            "artifact_score": 0.0,
            "temporal_score": 0.0,
            "metadata_score": 0.0,
            "compression_score": 0.0,
            "final_score": 0.1,
            "verdict": "REJECTED",
            "signals": ["no_file_provided"]
        }))
        sys.exit(0)  # Exit 0 so backend can parse the JSON verdict

    file_path = sys.argv[1]

    # Pre-check: does the file exist?
    if not os.path.exists(file_path):
        print(json.dumps({
            "model": "efficientnet+huggingface-ensemble",
            "model_score": 0.1,
            "artifact_score": 0.0,
            "temporal_score": 0.0,
            "metadata_score": 0.0,
            "compression_score": 0.0,
            "final_score": 0.1,
            "verdict": "REJECTED",
            "signals": [f"File {file_path} not found.", "fail_closed"]
        }))
        sys.exit(0)  # Exit 0 — backend will read the REJECTED verdict

    try:
        guard = DeepfakeGuardProcess()
        report = guard.run(file_path)
        print(json.dumps(report))
        sys.exit(0)
    except Exception as e:
        # Fail-closed: output valid JSON with REJECTED verdict
        # Exit 0 so the backend can still parse and use this decision
        print(json.dumps({
            "model": "efficientnet+huggingface-ensemble",
            "model_score": 0.1,
            "artifact_score": 0.0,
            "temporal_score": 0.0,
            "metadata_score": 0.0,
            "compression_score": 0.0,
            "final_score": 0.1,
            "verdict": "REJECTED",
            "signals": [f"engine_error: {str(e)}", "fail_closed"]
        }))
        sys.exit(0)
