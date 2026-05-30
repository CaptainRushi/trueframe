import os
import sys
import cv2
import numpy as np

# Safe imports for optional ML frameworks
_HAS_ONNX = False
try:
    import onnxruntime as ort
    _HAS_ONNX = True
except ImportError:
    pass

_HAS_TORCH = False
try:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    _HAS_TORCH = True
except ImportError:
    pass

def _log(msg):
    print(msg, file=sys.stderr)

class EfficientNetONNXDetector:
    def __init__(self, model_path):
        """
        Load the EfficientNet ONNX model using ONNX Runtime.
        """
        self.model_path = model_path
        self.session = None
        
        if not _HAS_ONNX:
            _log("[AI-MODEL] WARNING: ONNX Runtime not installed. EfficientNet ONNX cannot be used.")
            return

        if os.path.exists(model_path):
            loaded = False
            try:
                # Try CUDA first if available
                self.session = ort.InferenceSession(
                    model_path, 
                    providers=['CUDAExecutionProvider']
                )
                _log(f"[AI-MODEL] Loaded EfficientNet-B0 with CUDA from {model_path}")
                loaded = True
            except Exception as cuda_err:
                _log(f"[AI-MODEL] CUDA loading failed or unavailable: {cuda_err}. Trying CPU fallback.")
                
            if not loaded:
                try:
                    self.session = ort.InferenceSession(
                        model_path, 
                        providers=['CPUExecutionProvider']
                    )
                    _log(f"[AI-MODEL] Loaded EfficientNet-B0 with CPU from {model_path}")
                except Exception as cpu_err:
                    _log(f"[AI-MODEL] Error loading ONNX model on CPU: {cpu_err}")
        else:
            _log(f"[AI-MODEL] WARNING: Model file not found at {model_path}")

    def preprocess(self, face_crop_bgr):
        # 1. BGR to RGB
        img = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        # 2. Resize to 224x224
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        # 3. Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        # 4. ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        # 5. HWC -> CHW
        img = img.transpose(2, 0, 1)
        # 6. Add batch dimension: NCHW
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, face_crop_bgr):
        if self.session is None:
            # Raise model missing error if session is not loaded
            raise RuntimeError("EfficientNet ONNX model is missing. Failing closed.")
            
        try:
            input_tensor = self.preprocess(face_crop_bgr)
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            score = float(outputs[0].flatten()[0])
            return np.clip(score, 0.0, 1.0)
        except Exception as e:
            _log(f"[AI-MODEL] Inference error: {e}")
            raise RuntimeError(f"EfficientNet inference failed: {e}")

    def predict_batch(self, face_crops_bgr):
        if not face_crops_bgr:
            return []
        if self.session is None:
            raise RuntimeError("EfficientNet ONNX model is missing. Failing closed.")
            
        try:
            batch_tensors = [self.preprocess(crop) for crop in face_crops_bgr]
            input_batch = np.concatenate(batch_tensors, axis=0)
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_batch})
            scores = outputs[0].flatten().tolist()
            return [np.clip(s, 0.0, 1.0) for s in scores]
        except Exception as e:
            _log(f"[AI-MODEL] Batch inference error: {e}")
            raise RuntimeError(f"EfficientNet batch inference failed: {e}")

class SwinLONNXDetector:
    def __init__(self, model_path, fake_index=1):
        """
        Load the Swin-L ONNX model for tertiary review.
        """
        self.model_path = model_path
        self.fake_index = fake_index
        self.session = None
        self.channel_first = True
        self.input_size = (224, 224)

        if not _HAS_ONNX:
            _log("[AI-MODEL] WARNING: ONNX Runtime not installed. Swin-L cannot be used.")
            return

        if os.path.exists(model_path):
            loaded = False
            try:
                self.session = ort.InferenceSession(
                    model_path,
                    providers=['CUDAExecutionProvider']
                )
                _log(f"[AI-MODEL] Loaded Swin-L with CUDA from {model_path}")
                loaded = True
            except Exception as cuda_err:
                _log(f"[AI-MODEL] Swin-L CUDA loading failed: {cuda_err}. Trying CPU fallback.")

            if not loaded:
                try:
                    self.session = ort.InferenceSession(
                        model_path,
                        providers=['CPUExecutionProvider']
                    )
                    _log(f"[AI-MODEL] Loaded Swin-L with CPU from {model_path}")
                except Exception as cpu_err:
                    _log(f"[AI-MODEL] Error loading Swin-L ONNX model on CPU: {cpu_err}")
        else:
            _log(f"[AI-MODEL] WARNING: Swin-L model file not found at {model_path}")

        if self.session is not None:
            self._resolve_input_shape()

    def _resolve_input_shape(self):
        try:
            shape = self.session.get_inputs()[0].shape
            if len(shape) == 4:
                c_first = isinstance(shape[1], int) and shape[1] in (1, 3)
                c_last = isinstance(shape[3], int) and shape[3] in (1, 3)
                if c_first:
                    self.channel_first = True
                    h, w = shape[2], shape[3]
                elif c_last:
                    self.channel_first = False
                    h, w = shape[1], shape[2]
                else:
                    h, w = shape[2], shape[3]
                if isinstance(h, int) and isinstance(w, int):
                    self.input_size = (int(w), int(h))
        except Exception as e:
            _log(f"[AI-MODEL] Swin-L input shape detection failed: {e}")

    def preprocess(self, face_crop_bgr):
        img = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        if self.channel_first:
            img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def _softmax(self, logits):
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / (np.sum(exp) + 1e-9)

    def _extract_score(self, output):
        arr = np.array(output)
        if arr.ndim == 0:
            return float(arr)
        if arr.ndim == 1:
            if arr.shape[0] == 1:
                return float(arr[0])
            probs = self._softmax(arr)
            idx = min(max(self.fake_index, 0), probs.shape[0] - 1)
            return float(probs[idx])
        vec = arr[0]
        if vec.ndim == 0:
            return float(vec)
        if vec.shape[0] == 1:
            return float(vec[0])
        probs = self._softmax(vec)
        idx = min(max(self.fake_index, 0), probs.shape[0] - 1)
        return float(probs[idx])

    def predict(self, face_crop_bgr):
        if self.session is None:
            _log("[AI-MODEL] Swin-L unavailable. Skipping tertiary review.")
            return None

        try:
            input_tensor = self.preprocess(face_crop_bgr)
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            score = self._extract_score(outputs[0])
            return float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            _log(f"[AI-MODEL] Swin-L inference error: {e}")
            return None

class HuggingFaceDeepfakeDetector:
    # Known label maps for specific models where autodetect fails.
    # Key = model name, Value = index of "fake" class in softmax output.
    KNOWN_FAKE_INDICES = {
        "dima806/deepfake_vs_real_image_detection": 1,   # id2label: {0: "Real", 1: "Fake"}
        "prithivMLmods/Deep-Fake-Detector-v2-Model": 1,  # id2label: {0: "Realism", 1: "Deepfake"}
    }

    def __init__(self, model_name="dima806/deepfake_vs_real_image_detection"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.fake_idx = 0

        if not _HAS_TORCH:
            _log("[AI-MODEL] PyTorch/Transformers not installed. HuggingFaceDeepfakeDetector will use signal fallback.")
            return

        try:
            _log(f"[AI-MODEL] Loading HuggingFace model: {model_name}")
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()

            # ── Label-index resolution ──────────────────────────────────────
            # 1. Check the known override table first (most reliable).
            if model_name in self.KNOWN_FAKE_INDICES:
                self.fake_idx = self.KNOWN_FAKE_INDICES[model_name]
                _log(f"[AI-MODEL] Using known fake index={self.fake_idx} for '{model_name}'")
            else:
                # 2. Auto-detect from id2label: search for "fake" keyword.
                detected_idx = None
                label_map = {}
                for idx, label in self.model.config.id2label.items():
                    label_map[int(idx)] = label
                    if "fake" in label.lower() or "forged" in label.lower() or "manipulat" in label.lower():
                        detected_idx = int(idx)
                _log(f"[AI-MODEL] id2label map: {label_map}")
                if detected_idx is not None:
                    self.fake_idx = detected_idx
                # else: keep default fake_idx=0

            _log(f"[AI-MODEL] HuggingFace loaded on {self.device}. Fake label mapped to index: {self.fake_idx}")
        except Exception as e:
            _log(f"[AI-MODEL] Error loading HuggingFace model: {e}")

    def preprocess(self, face_crop_bgr):
        img_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    def predict(self, face_crop_bgr):
        if not _HAS_TORCH or self.model is None or self.processor is None:
            # Signal-based fallback: analyze texture and blending edges
            try:
                gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                too_smooth = lap_var < 80.0
                
                h, w = gray.shape
                bw = max(4, int(min(h, w) * 0.1))
                border = gray[:bw, :].mean() + gray[-bw:, :].mean() + gray[:, :bw].mean() + gray[:, -bw:].mean()
                border /= 4.0
                interior = gray[bw:-bw, bw:-bw].mean()
                diff = abs(border - interior)
                seam_detected = diff > 25.0

                if too_smooth or seam_detected:
                    return 0.85 # Suspected fake
                return 0.15 # Suspected real
            except Exception:
                return 0.2

        try:
            pil_img = self.preprocess(face_crop_bgr)
            inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = outputs.logits.softmax(dim=1)
                score = probs[0][self.fake_idx].item()
            return np.clip(score, 0.0, 1.0)
        except Exception as e:
            _log(f"[AI-MODEL] HF Inference error: {e}")
            return 0.1

    def predict_batch(self, face_crops_bgr):
        if not face_crops_bgr:
            return []
        if not _HAS_TORCH or self.model is None or self.processor is None:
            return [self.predict(crop) for crop in face_crops_bgr]
            
        try:
            imgs = [self.preprocess(crop) for crop in face_crops_bgr]
            inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = outputs.logits.softmax(dim=1)
                scores = probs[:, self.fake_idx].tolist()
            return [np.clip(s, 0.0, 1.0) for s in scores]
        except Exception as e:
            _log(f"[AI-MODEL] HF Batch inference error: {e}")
            return [0.1] * len(face_crops_bgr)
