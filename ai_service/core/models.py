import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

class EfficientNetONNXDetector:
    def __init__(self, model_path):
        """
        Load the EfficientNet ONNX model using ONNX Runtime with CPUExecutionProvider only.
        """
        self.model_path = model_path
        self.session = None
        
        if os.path.exists(model_path):
            try:
                # Use CPUExecutionProvider only as per requirement
                self.session = ort.InferenceSession(
                    model_path, 
                    providers=['CPUExecutionProvider']
                )
                print(f"[AI-MODEL] Loaded EfficientNet-B0 from {model_path}")
            except Exception as e:
                print(f"[AI-MODEL] Error loading ONNX model: {e}")
        else:
            print(f"[AI-MODEL] WARNING: Model file not found at {model_path}")

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
            # FAIL-CLOSED: If model missing or error, return high risk (1.0)
            return 1.0
            
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
            print(f"[AI-MODEL] Inference error: {e}")
            return 1.0 # FAIL-CLOSED

    def predict_batch(self, face_crops_bgr):
        """
        Use batch inference when processing multiple frames for efficiency.
        """
        if not face_crops_bgr or self.session is None:
            return [1.0] * len(face_crops_bgr) if face_crops_bgr else []
            
        try:
            # Batch preprocessing
            batch_tensors = [self.preprocess(crop) for crop in face_crops_bgr]
            input_batch = np.concatenate(batch_tensors, axis=0)
            
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_batch})
            
            scores = outputs[0].flatten().tolist()
            return [np.clip(s, 0.0, 1.0) for s in scores]
            
        except Exception as e:
            print(f"[AI-MODEL] Batch inference error: {e}")
            return [1.0] * len(face_crops_bgr)
