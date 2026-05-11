"""
TrueFrame Reels — Production Inference Engine
===============================================
Real-time reel deepfake detection for the TrueFrame platform.
Integrates the trained EfficientNet-B4+LSTM model with the
existing ai_service verification pipeline.

Usage:
    python reel_inference.py <video_path>

Output: JSON with deepfake probability, verdict, and signals.
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


class ReelInferenceEngine:
    """
    Production inference for reel deepfake detection.
    Supports both ONNX and TorchScript backends.
    """

    def __init__(
        self,
        model_path: str = None,
        backend: str = "onnx",
        max_frames: int = 15,
        device: str = "auto",
    ):
        self.max_frames = max_frames
        self.frame_size = (224, 224)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Decision thresholds (aligned with TrueFrame system)
        self.THRESHOLD_APPROVE = 0.40
        self.THRESHOLD_REVIEW = 0.60
        self.THRESHOLD_REJECT = 0.80

        # Load model
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models"
        )

        if backend == "onnx":
            self._load_onnx(
                model_path
                or os.path.join(models_dir, "trueframe_reels_detector.onnx")
            )
        else:
            self._load_torchscript(
                model_path
                or os.path.join(models_dir, "trueframe_reels_detector.pt")
            )

        # Face detector
        self._init_face_detector()
        self.backend = backend

    def _load_onnx(self, path):
        try:
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            try:
                if ort.get_device() == "GPU":
                    providers.insert(0, "CUDAExecutionProvider")
            except Exception:
                pass

            self.session = ort.InferenceSession(path, providers=providers)
            self.model_type = "onnx"
            logger.info(f"ONNX model loaded: {path}")
        except Exception as e:
            logger.warning(f"ONNX load failed: {e}")
            self.session = None
            self.model_type = "fallback"

    def _load_torchscript(self, path):
        try:
            import torch

            self.model = torch.jit.load(path)
            self.model.eval()
            self.model_type = "torchscript"
            logger.info(f"TorchScript model loaded: {path}")
        except Exception as e:
            logger.warning(f"TorchScript load failed: {e}")
            self.model = None
            self.model_type = "fallback"

    def _init_face_detector(self):
        try:
            import mediapipe as mp
            try:
                import mediapipe.python.solutions as mp_solutions
                self.mp_face = mp_solutions.face_detection
            except (ImportError, AttributeError):
                self.mp_face = mp.solutions.face_detection

            self.face_det = self.mp_face.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
            self.face_backend = "mediapipe"
        except Exception:
            self.face_det = cv2.CascadeClassifier(
                cv2.data.haarcascades
                + "haarcascade_frontalface_default.xml"
            )
            self.face_backend = "haar"

    def analyze_reel(self, video_path: str) -> dict:
        """
        Analyze a reel video for deepfake content.
        Returns JSON-compatible dict with scores and verdict.
        """
        start_time = time.time()
        signals = []

        # 1. Extract frames
        face_crops = self._extract_face_sequence(video_path)
        if face_crops is None or len(face_crops) < 3:
            return self._build_result(
                0.5, ["insufficient_face_data"], start_time
            )

        # 2. Preprocess
        input_tensor = self._preprocess_sequence(face_crops)

        # 3. Inference
        deepfake_prob = self._predict(input_tensor)

        if deepfake_prob is None:
            return self._build_result(
                0.5, ["model_inference_failed"], start_time
            )

        # 4. Build result
        return self._build_result(deepfake_prob, signals, start_time)

    def _extract_face_sequence(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps if fps > 0 else 0

        if duration <= 0:
            cap.release()
            return None

        # Sample timestamps
        n = min(self.max_frames, max(10, int(duration * 2)))
        times = np.linspace(0.5, duration - 0.5, n)

        crops = []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            face = self._detect_face(frame)
            if face is not None:
                face = cv2.resize(face, self.frame_size)
                crops.append(face)

        cap.release()

        if len(crops) < 3:
            return None

        # Normalize to exactly 10 frames
        target = 10
        if len(crops) >= target:
            indices = np.linspace(0, len(crops) - 1, target, dtype=int)
            crops = [crops[i] for i in indices]
        else:
            while len(crops) < target:
                crops.append(crops[-1].copy())

        return crops

    def _detect_face(self, frame):
        if self.face_backend == "mediapipe":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_det.process(rgb)
            if not results.detections:
                return None
            best = max(results.detections, key=lambda d: d.score[0])
            bbox = best.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, x1 + int(bbox.width * w))
            y2 = min(h, y1 + int(bbox.height * h))
            if x2 - x1 < 32 or y2 - y1 < 32:
                return None
            return frame[y1:y2, x1:x2]
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_det.detectMultiScale(gray, 1.1, 4)
            if len(faces) == 0:
                return None
            x, y, w, h = sorted(
                faces, key=lambda f: f[2] * f[3], reverse=True
            )[0]
            return frame[y : y + h, x : x + w]

    def _preprocess_sequence(self, crops):
        processed = []
        for crop in crops:
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
            img /= 255.0
            img = (img - self.mean) / self.std
            img = img.transpose(2, 0, 1)  # HWC → CHW
            processed.append(img)

        # (1, T, 3, H, W)
        return np.expand_dims(np.stack(processed), axis=0).astype(
            np.float32
        )

    def _predict(self, input_tensor):
        if self.model_type == "onnx" and self.session:
            try:
                name = self.session.get_inputs()[0].name
                out = self.session.run(None, {name: input_tensor})
                return float(np.clip(out[0].flatten()[0], 0, 1))
            except Exception as e:
                logger.error(f"ONNX inference error: {e}")
                return None

        elif self.model_type == "torchscript" and self.model:
            try:
                import torch

                t = torch.from_numpy(input_tensor)
                with torch.no_grad():
                    out = self.model(t)
                return float(out.flatten()[0].item())
            except Exception as e:
                logger.error(f"TorchScript inference error: {e}")
                return None

        return None

    def _build_result(self, prob, signals, start_time):
        elapsed_ms = (time.time() - start_time) * 1000
        prob_rounded = round(prob, 4)
        authenticity_score = round(1.0 - prob, 4)

        if prob >= self.THRESHOLD_REJECT:
            verdict = "REJECTED"
            confidence = "HIGH"
            signals.append("deepfake_detected")
        elif prob >= self.THRESHOLD_REVIEW:
            verdict = "UNDER_REVIEW"
            confidence = "MEDIUM"
            signals.append("borderline_needs_review")
        elif prob >= self.THRESHOLD_APPROVE:
            verdict = "UNDER_REVIEW"
            confidence = "LOW"
        else:
            verdict = "APPROVED"
            confidence = "HIGH"

        return {
            "model": "trueframe-reels-efficientnet-lstm",
            "model_score": prob_rounded,
            "artifact_score": 0.0,
            "temporal_score": 0.0,
            "metadata_score": 0.0,
            "compression_score": 0.0,
            "final_score": prob_rounded,
            "deepfake_probability": prob_rounded,
            "authenticity_score": authenticity_score,
            "verdict": verdict,
            "confidence": confidence,
            "inference_time_ms": round(elapsed_ms, 1),
            "signals": list(set(signals)),
        }


# ────────────── CLI ENTRY POINT ──────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print(
            json.dumps(
                {
                    "model": "trueframe-reels-efficientnet-lstm",
                    "model_score": 0.5,
                    "artifact_score": 0.0,
                    "temporal_score": 0.0,
                    "metadata_score": 0.0,
                    "compression_score": 0.0,
                    "final_score": 0.5,
                    "deepfake_probability": 0.5,
                    "authenticity_score": 0.5,
                    "verdict": "REJECTED",
                    "signals": ["no_file_provided"],
                }
            )
        )
        sys.exit(1)

    try:
        engine = ReelInferenceEngine()
        result = engine.analyze_reel(sys.argv[1])
        print(json.dumps(result))
    except Exception as e:
        print(
            json.dumps(
                {
                    "model": "trueframe-reels-efficientnet-lstm",
                    "model_score": 0.5,
                    "artifact_score": 0.0,
                    "temporal_score": 0.0,
                    "metadata_score": 0.0,
                    "compression_score": 0.0,
                    "final_score": 0.5,
                    "deepfake_probability": 0.5,
                    "authenticity_score": 0.5,
                    "verdict": "REJECTED",
                    "signals": [f"engine_error: {str(e)}", "fail_closed"],
                }
            )
        )
        sys.exit(1)
