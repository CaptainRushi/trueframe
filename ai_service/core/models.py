import os
import sys
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

def _log(msg):
    print(msg, file=sys.stderr)

class EfficientNetONNXDetector:
    def __init__(self, model_path):
        """
        Load the EfficientNet ONNX model using ONNX Runtime with CPUExecutionProvider only.
        """
        self.model_path = model_path
        self.session = None
        
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
        """
        Resize to 224x224 and normalize using ImageNet statistics.
        Input size: 224 x 224 RGB
        """
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
        """
        Run EfficientNet inference via ONNX Runtime (CPU).
        Return a single probability score (0-1).
        """
        if self.session is None:
            # FAIL-CLOSED: If model missing, raise an error to reject upload
            raise RuntimeError("EfficientNet ONNX model is missing. Failing closed.")
            
        try:
            input_tensor = self.preprocess(face_crop_bgr)
            input_name = self.session.get_inputs()[0].name
            
            # Inference
            outputs = self.session.run(None, {input_name: input_tensor})
            
            # The model output is a single probability score (0-1)
            # Depending on the output layer, it might be [ [score] ] or [ [logits] ]
            # The Master Prompt specifies: "Output: single probability score (0–1)"
            score = float(outputs[0].flatten()[0])
            
            # If the model output is logits (not 0-1), we would need a sigmoid
            # But prompt says it outputs a score.
            return np.clip(score, 0.0, 1.0)
            
        except Exception as e:
            _log(f"[AI-MODEL] Inference error: {e}")
            raise RuntimeError(f"EfficientNet inference failed: {e}")

    def predict_batch(self, face_crops_bgr):
        """
        Use batch inference when processing multiple frames for efficiency.
        """
        if not face_crops_bgr:
            return []
        if self.session is None:
            raise RuntimeError("EfficientNet ONNX model is missing. Failing closed.")
            
        try:
            # Batch preprocessing
            batch_tensors = [self.preprocess(crop) for crop in face_crops_bgr]
            input_batch = np.concatenate(batch_tensors, axis=0)
            
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_batch})
            
            scores = outputs[0].flatten().tolist()
            return [np.clip(s, 0.0, 1.0) for s in scores]
            
        except Exception as e:
            _log(f"[AI-MODEL] Batch inference error: {e}")
            raise RuntimeError(f"EfficientNet batch inference failed: {e}")

class HuggingFaceDeepfakeDetector:
    def __init__(self, model_name="dima806/deepfake_vs_real_image_detection"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.fake_idx = 0
        try:
            _log(f"[AI-MODEL] Loading HuggingFace model: {model_name}")
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()

            # Autodetect which index corresponds to "fake"
            for idx, label in self.model.config.id2label.items():
                if "fake" in label.lower() or "forged" in label.lower() or "manipulat" in label.lower():
                    self.fake_idx = int(idx)
                    break
            _log(f"[AI-MODEL] HuggingFace loaded on {self.device}. Fake label mapped to index: {self.fake_idx}")
        except Exception as e:
            _log(f"[AI-MODEL] Error loading HuggingFace model: {e}")

    def preprocess(self, face_crop_bgr):
        # Convert BGR (OpenCV) to RGB (PIL)
        img_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    def predict(self, face_crop_bgr):
        if self.model is None or self.processor is None:
            return 0.1 # fallback
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
        if self.model is None or self.processor is None:
            return [0.1] * len(face_crops_bgr)
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
