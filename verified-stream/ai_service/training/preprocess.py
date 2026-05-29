"""
TrueFrame — Frame Preprocessing Pipeline
==========================================
Implements the novel preprocessing strategy from the LightFakeDetect paper:
  1. Extract frames uniformly from video
  2. Filter near-duplicate frames using SSIM (instead of random removal)
  3. Detect and crop face regions via MTCNN (with Haar cascade fallback)
  4. Resize to 224×224 and normalize (ImageNet mean/std)

Key innovation: similarity-based deduplication preserves informative temporal
content while removing redundant frames, giving the GRU more useful signal.

Reference:
  "LightFakeDetect: A Lightweight Deepfake Video Detection Architecture"
  MDPI Applied Sciences, 2024.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional

# ImageNet normalization constants (used by MobileNetV2 pretrained on ImageNet)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FRAME_SIZE    = (224, 224)
MAX_FRAMES    = 20      # Maximum frames to keep after deduplication
SSIM_THRESHOLD = 0.95   # Frames more similar than this are considered duplicates


# ─────────────────── SSIM HELPERS ────────────────────

def _ssim_grayscale(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two grayscale images.
    Returns float in [0, 1] — higher means more similar.
    Avoids scikit-image import when possible; uses fast NumPy implementation.
    """
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(img1, img2, data_range=255.0))
    except ImportError:
        pass

    # Fallback: fast luminance-based similarity (approximates SSIM)
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = img1.std()
    sigma2 = img2.std()
    sigma12 = float(np.mean((img1 - mu1) * (img2 - mu2)))
    C1, C2 = 6.5025, 58.5225
    numerator   = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2)
    return float(numerator / (denominator + 1e-8))


def filter_similar_frames(
    frames: List[np.ndarray],
    ssim_threshold: float = SSIM_THRESHOLD,
    max_frames: int = MAX_FRAMES,
) -> List[np.ndarray]:
    """
    Remove near-duplicate frames using SSIM similarity.

    The paper's key insight: random frame removal wastes temporal diversity.
    By comparing consecutive frames and skipping ones that are too similar,
    we keep frames that actually contain new information.

    Args:
        frames:          List of BGR frames (any size)
        ssim_threshold:  SSIM score above which a frame is considered a duplicate
        max_frames:      Hard cap on returned frames

    Returns:
        Filtered list of unique, informative frames
    """
    if not frames:
        return []

    # Resize to small thumbnail for fast SSIM comparison
    thumb_size = (64, 64)
    filtered  = [frames[0]]
    prev_gray  = cv2.resize(
        cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), thumb_size
    ).astype(np.float32)

    for frame in frames[1:]:
        gray = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), thumb_size
        ).astype(np.float32)

        similarity = _ssim_grayscale(prev_gray, gray)

        if similarity < ssim_threshold:
            # Frame is sufficiently different → keep it
            filtered.append(frame)
            prev_gray = gray

        if len(filtered) >= max_frames:
            break

    return filtered


# ─────────────────── FACE DETECTION ──────────────────

def _build_mtcnn_detector():
    """
    Build MTCNN face detector from facenet-pytorch.
    Returns a callable detect_fn(frame_bgr) → Optional[np.ndarray face_crop_bgr].
    Falls back to MediaPipe → Haar cascade if not available.
    """
    try:
        from facenet_pytorch import MTCNN
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mtcnn = MTCNN(
            image_size=224,
            margin=20,           # Extra pixels around face bounding box
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net thresholds
            factor=0.709,
            post_process=False,  # Return PIL images (not tensors)
            device=device,
            keep_all=False,      # Only the most prominent face
        )

        def detect_mtcnn(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
            from PIL import Image
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            try:
                # MTCNN returns cropped face as numpy array (224×224 already)
                face = mtcnn(pil)
                if face is None:
                    return None
                # face is a torch tensor (3, 224, 224) in [0, 255]
                face_np = face.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                return face_bgr
            except Exception:
                return None

        return detect_mtcnn

    except ImportError:
        pass  # MTCNN unavailable → fall through to MediaPipe

    try:
        import mediapipe as mp
        try:
            face_det = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.4
            )
        except Exception:
            face_det = mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=0.4
            )

        def detect_mediapipe(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
            rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = face_det.process(rgb)
            if not results.detections:
                return None
            best = max(results.detections, key=lambda d: d.score[0])
            bbox = best.location_data.relative_bounding_box
            fh, fw = frame_bgr.shape[:2]
            pad_x = bbox.width  * 0.15
            pad_y = bbox.height * 0.15
            x1 = max(0, int((bbox.xmin - pad_x) * fw))
            y1 = max(0, int((bbox.ymin - pad_y) * fh))
            x2 = min(fw, int((bbox.xmin + bbox.width  + pad_x) * fw))
            y2 = min(fh, int((bbox.ymin + bbox.height + pad_y) * fh))
            if x2 - x1 < 32 or y2 - y1 < 32:
                return None
            return frame_bgr[y1:y2, x1:x2]

        return detect_mediapipe

    except ImportError:
        pass  # MediaPipe unavailable → Haar cascade

    def detect_haar(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        for neighbors in [3, 2]:
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=neighbors, minSize=(32, 32)
            )
            if len(faces) > 0:
                x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                pad = int(min(w, h) * 0.15)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame_bgr.shape[1], x + w + pad)
                y2 = min(frame_bgr.shape[0], y + h + pad)
                return frame_bgr[y1:y2, x1:x2]
        return None

    return detect_haar


# Singleton detector (lazy init on first call)
_DETECTOR = None

def _get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = _build_mtcnn_detector()
    return _DETECTOR


# ─────────────────── NORMALIZATION ───────────────────

def normalize_frame(face_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR face crop (uint8, 224×224) to normalized float32 tensor.
    Applies ImageNet mean/std normalization as expected by MobileNetV2.

    Returns: np.ndarray of shape (3, 224, 224), dtype float32.
    """
    # Resize to FRAME_SIZE if needed
    if face_bgr.shape[:2] != FRAME_SIZE:
        face_bgr = cv2.resize(face_bgr, FRAME_SIZE)

    # BGR → RGB, normalize to [0, 1]
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Apply ImageNet normalization
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD  # (224, 224, 3)

    # HWC → CHW (PyTorch convention)
    return normalized.transpose(2, 0, 1)   # (3, 224, 224)


# ─────────────────── FRAME EXTRACTION ────────────────

def extract_frames(video_path: str, n: int = MAX_FRAMES * 3) -> List[np.ndarray]:
    """
    Uniformly sample up to n frames from a video.
    Samples 3× the target count before SSIM filtering to ensure enough remain.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / max(fps, 1.0)
    times = np.linspace(0.5, max(0.5, duration - 0.5), n)
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
    cap.release()
    return frames


# ─────────────────── FULL PIPELINE ───────────────────

def preprocess_video(
    video_path: str,
    max_frames: int = MAX_FRAMES,
    ssim_threshold: float = SSIM_THRESHOLD,
) -> Tuple[List[np.ndarray], bool]:
    """
    Full preprocessing pipeline for a video.

    Steps:
        1. Extract frames uniformly (3× target count for SSIM budget)
        2. Filter near-duplicate frames via SSIM
        3. Detect and crop face regions (MTCNN → MediaPipe → Haar)
        4. Resize crops to 224×224
        5. Return list of raw BGR face crops (normalization done in inference)

    Args:
        video_path:     Path to video file
        max_frames:     Target number of frames to return
        ssim_threshold: SSIM above which frames are considered duplicates

    Returns:
        (face_crops_bgr, has_faces):
            face_crops_bgr — list of (224, 224, 3) BGR arrays
            has_faces      — True if at least one face was detected
    """
    # 1. Extract frames
    raw_frames = extract_frames(video_path, n=max_frames * 3)
    if not raw_frames:
        return [], False

    # 2. SSIM-based deduplication
    unique_frames = filter_similar_frames(raw_frames, ssim_threshold, max_frames * 2)
    if not unique_frames:
        unique_frames = raw_frames[:max_frames]

    # 3. Face detection and crop
    detector = _get_detector()
    face_crops = []

    for frame in unique_frames:
        face = detector(frame)
        if face is None:
            # Try on brightened version for dark frames
            bright = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)
            face = detector(bright)
        if face is not None and face.size > 0:
            face_resized = cv2.resize(face, FRAME_SIZE)
            face_crops.append(face_resized)

        if len(face_crops) >= max_frames:
            break

    has_faces = len(face_crops) >= 1
    return face_crops, has_faces


def preprocess_image(image_path: str) -> Tuple[Optional[np.ndarray], bool]:
    """
    Full preprocessing pipeline for a single image.

    Args:
        image_path: Path to image file

    Returns:
        (face_crop_bgr, has_face):
            face_crop_bgr — (224, 224, 3) BGR array or None
            has_face      — True if a face was detected
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, False

    detector = _get_detector()
    face = detector(img)

    if face is None:
        # Try brightened
        bright = cv2.convertScaleAbs(img, alpha=1.3, beta=20)
        face = detector(bright)

    if face is not None and face.size > 0:
        face_resized = cv2.resize(face, FRAME_SIZE)
        return face_resized, True

    return None, False


def crops_to_tensor(face_crops: List[np.ndarray]) -> Optional["torch.Tensor"]:
    """
    Convert list of BGR face crops to a normalized PyTorch tensor.

    Args:
        face_crops: List of (224, 224, 3) BGR uint8 arrays

    Returns:
        torch.Tensor of shape (1, T, 3, 224, 224) — batch=1, T=seq_len
        or None if face_crops is empty
    """
    try:
        import torch
        if not face_crops:
            return None
        normalized = [normalize_frame(c) for c in face_crops]    # list of (3, 224, 224)
        arr = np.stack(normalized, axis=0)                         # (T, 3, 224, 224)
        tensor = torch.from_numpy(arr).unsqueeze(0).float()        # (1, T, 3, 224, 224)
        return tensor
    except ImportError:
        return None
