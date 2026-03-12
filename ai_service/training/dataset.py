"""
TrueFrame Reels — Dataset & Data Loading
==========================================
Handles loading from FaceForensics++, DFDC, Celeb-DF, and custom datasets.
Performs frame extraction, face detection, sequence construction,
and augmentation for the EfficientNet-B4 + LSTM training pipeline.
"""

import os
import sys
import cv2
import json
import random
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Add parent for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.config import CONFIG
from training.augmentations import get_train_transforms, get_val_transforms

logger = logging.getLogger("trueframe.dataset")


# ─────────────────────── FACE DETECTOR ───────────────

class ReelsFaceDetector:
    """
    Lightweight face detector for training data preprocessing.
    Uses MediaPipe (preferred) or falls back to Haar cascades.
    """

    def __init__(self, confidence: float = 0.5):
        self.confidence = confidence
        self._init_detector()

    def _init_detector(self):
        try:
            import mediapipe as mp
            try:
                import mediapipe.python.solutions as mp_solutions
                self.mp_face = mp_solutions.face_detection
            except (ImportError, AttributeError):
                self.mp_face = mp.solutions.face_detection

            self.detector = self.mp_face.FaceDetection(
                model_selection=1,  # Full-range model for varied distances
                min_detection_confidence=self.confidence,
            )
            self.backend = "mediapipe"
        except Exception:
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self.backend = "haar"

    def detect_and_crop(self, frame: np.ndarray, margin: float = 0.2) -> Optional[np.ndarray]:
        """Detect the largest face and return a cropped+padded region."""
        if self.backend == "mediapipe":
            return self._detect_mediapipe(frame, margin)
        return self._detect_haar(frame, margin)

    def _detect_mediapipe(self, frame, margin):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        if not results.detections:
            return None

        best = max(results.detections, key=lambda d: d.score[0])
        bbox = best.location_data.relative_bounding_box
        h, w = frame.shape[:2]

        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        # Add margin
        mx, my = int(bw * margin), int(bh * margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w, x + bw + mx)
        y2 = min(h, y + bh + my)

        if x2 - x1 < 32 or y2 - y1 < 32:
            return None

        return frame[y1:y2, x1:x2]

    def _detect_haar(self, frame, margin):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return None

        x, y, fw, fh = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        h, w = frame.shape[:2]
        mx, my = int(fw * margin), int(fh * margin)
        x1, y1 = max(0, x - mx), max(0, y - my)
        x2, y2 = min(w, x + fw + mx), min(h, y + fh + my)

        if x2 - x1 < 32 or y2 - y1 < 32:
            return None

        return frame[y1:y2, x1:x2]


# ─────────────────── FRAME EXTRACTION ────────────────

class ReelsFrameExtractor:
    """
    Extract and sample frames from short-form videos.
    Optimized for vertical reels (9:16, 5–60 seconds).
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or CONFIG.frames
        self.face_detector = ReelsFaceDetector(
            confidence=self.cfg.FACE_CONFIDENCE_THRESHOLD
        )

    def extract_face_sequence(
        self, video_path: str
    ) -> Tuple[Optional[List[np.ndarray]], dict]:
        """
        Extract a sequence of face crops from a video.
        Returns (face_crops, metadata) or (None, metadata) on failure.
        """
        meta = {"path": video_path, "frames_extracted": 0, "faces_detected": 0}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            meta["error"] = "cannot_open_video"
            return None, meta

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        meta["fps"] = fps
        meta["duration"] = duration

        if duration < self.cfg.MIN_DURATION_SEC or duration > self.cfg.MAX_DURATION_SEC:
            cap.release()
            meta["error"] = f"duration_out_of_range ({duration:.1f}s)"
            return None, meta

        # Determine sample timestamps
        sample_times = self._get_sample_times(duration)

        face_crops = []
        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                continue
            meta["frames_extracted"] += 1

            face = self.face_detector.detect_and_crop(frame)
            if face is not None:
                # Resize to target
                face = cv2.resize(
                    face,
                    self.cfg.FRAME_SIZE,
                    interpolation=cv2.INTER_LINEAR,
                )
                face_crops.append(face)
                meta["faces_detected"] += 1

        cap.release()

        if len(face_crops) < 3:  # Need minimum 3 frames for temporal analysis
            meta["error"] = "insufficient_face_crops"
            return None, meta

        # Pad or truncate to SEQUENCE_LENGTH
        face_crops = self._normalize_sequence(face_crops)
        return face_crops, meta

    def _get_sample_times(self, duration: float) -> List[float]:
        """Generate frame timestamps based on sampling strategy."""
        strategy = self.cfg.SAMPLING_STRATEGY
        n = min(self.cfg.MAX_FRAMES_PER_VIDEO, int(duration * self.cfg.SAMPLE_FPS))
        n = max(n, self.cfg.SEQUENCE_LENGTH)

        if strategy == "uniform":
            return np.linspace(0.5, duration - 0.5, n).tolist()
        elif strategy == "beginning_middle_end":
            # 30% beginning, 40% middle, 30% end
            beg = np.linspace(0.5, duration * 0.3, max(1, int(n * 0.3))).tolist()
            mid = np.linspace(duration * 0.3, duration * 0.7, max(1, int(n * 0.4))).tolist()
            end = np.linspace(duration * 0.7, duration - 0.5, max(1, int(n * 0.3))).tolist()
            return sorted(set(beg + mid + end))
        elif strategy == "keyframe":
            return np.linspace(0, duration - 0.5, n).tolist()
        else:
            return np.linspace(0.5, duration - 0.5, n).tolist()

    def _normalize_sequence(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """Ensure sequence is exactly SEQUENCE_LENGTH frames."""
        seq_len = self.cfg.SEQUENCE_LENGTH
        if len(crops) >= seq_len:
            # Uniformly sample seq_len frames
            indices = np.linspace(0, len(crops) - 1, seq_len, dtype=int)
            return [crops[i] for i in indices]
        else:
            # Pad by repeating last frame
            while len(crops) < seq_len:
                crops.append(crops[-1].copy())
            return crops


# ─────────────── DATASET REGISTRY / LOADERS ──────────

class FaceForensicsLoader:
    """
    Loads video paths + labels from FaceForensics++ directory structure.
    Expected layout:
        FaceForensics/
        ├── original_sequences/youtube/c23/videos/
        ├── manipulated_sequences/Deepfakes/c23/videos/
        ├── manipulated_sequences/Face2Face/c23/videos/
        ├── manipulated_sequences/FaceSwap/c23/videos/
        └── manipulated_sequences/NeuralTextures/c23/videos/
    """

    MANIPULATION_MAP = {
        "Deepfakes": "face_swap",
        "Face2Face": "face_reenactment",
        "FaceSwap": "face_swap",
        "NeuralTextures": "neural_texture",
    }

    @staticmethod
    def load(root: str, compression: str = "c23") -> List[Dict[str, Any]]:
        entries = []
        real_dir = os.path.join(root, "original_sequences", "youtube", compression, "videos")
        if os.path.isdir(real_dir):
            for f in os.listdir(real_dir):
                if f.endswith((".mp4", ".avi")):
                    entries.append({
                        "path": os.path.join(real_dir, f),
                        "label": 0,                  # Real
                        "manipulation_type": "real",
                        "source": "faceforensics",
                    })

        for manip, mtype in FaceForensicsLoader.MANIPULATION_MAP.items():
            fake_dir = os.path.join(root, "manipulated_sequences", manip, compression, "videos")
            if os.path.isdir(fake_dir):
                for f in os.listdir(fake_dir):
                    if f.endswith((".mp4", ".avi")):
                        entries.append({
                            "path": os.path.join(fake_dir, f),
                            "label": 1,              # Deepfake
                            "manipulation_type": mtype,
                            "source": "faceforensics",
                        })
        return entries


class DFDCLoader:
    """
    Loads from DFDC dataset.
    Expected layout:
        DFDC/
        ├── dfdc_train_part_0/
        │   ├── *.mp4
        │   └── metadata.json
        ├── dfdc_train_part_1/ ...
    """

    @staticmethod
    def load(root: str) -> List[Dict[str, Any]]:
        entries = []
        for part_dir in sorted(Path(root).glob("dfdc_train_part_*")):
            meta_file = part_dir / "metadata.json"
            if not meta_file.exists():
                continue
            with open(meta_file) as f:
                metadata = json.load(f)
            for fname, info in metadata.items():
                fpath = part_dir / fname
                if fpath.exists():
                    label = 1 if info.get("label", "REAL") == "FAKE" else 0
                    entries.append({
                        "path": str(fpath),
                        "label": label,
                        "manipulation_type": "face_swap" if label == 1 else "real",
                        "source": "dfdc",
                    })
        return entries


class CelebDFLoader:
    """
    Loads from Celeb-DF v2 dataset.
    Expected layout:
        CelebDF/
        ├── Celeb-real/ (*.mp4)
        ├── Celeb-synthesis/ (*.mp4)
        ├── YouTube-real/ (*.mp4)
        └── List_of_testing_videos.txt
    """

    @staticmethod
    def load(root: str) -> List[Dict[str, Any]]:
        entries = []
        for subdir, label, mtype in [
            ("Celeb-real", 0, "real"),
            ("YouTube-real", 0, "real"),
            ("Celeb-synthesis", 1, "face_swap"),
        ]:
            d = os.path.join(root, subdir)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.endswith((".mp4", ".avi")):
                        entries.append({
                            "path": os.path.join(d, f),
                            "label": label,
                            "manipulation_type": mtype,
                            "source": "celeb_df",
                        })
        return entries


class CustomReelsLoader:
    """
    Loads from a custom directory with real/ and fake/ sub-folders.
    Expected layout:
        custom_reels/
        ├── real/ (*.mp4)
        └── fake/ (*.mp4)
    """

    @staticmethod
    def load(root: str) -> List[Dict[str, Any]]:
        entries = []
        for subdir, label, mtype in [
            ("real", 0, "real"),
            ("fake", 1, "face_swap"),
        ]:
            d = os.path.join(root, subdir)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.endswith((".mp4", ".avi", ".mov", ".webm")):
                        entries.append({
                            "path": os.path.join(d, f),
                            "label": label,
                            "manipulation_type": mtype,
                            "source": "custom",
                        })
        return entries


# ──────────────── COMBINED REGISTRY ──────────────────

DATASET_LOADERS = {
    "faceforensics": lambda: FaceForensicsLoader.load(CONFIG.dataset.FACEFORENSICS_ROOT),
    "dfdc": lambda: DFDCLoader.load(CONFIG.dataset.DFDC_ROOT),
    "celeb_df": lambda: CelebDFLoader.load(CONFIG.dataset.CELEB_DF_ROOT),
    "custom": lambda: CustomReelsLoader.load(CONFIG.dataset.CUSTOM_ROOT),
}


def load_all_entries(datasets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Load and merge video entries from specified (or all available) datasets."""
    if datasets is None:
        datasets = CONFIG.dataset.SUPPORTED_DATASETS

    all_entries = []
    for ds_name in datasets:
        loader = DATASET_LOADERS.get(ds_name)
        if loader is None:
            logger.warning(f"Unknown dataset: {ds_name}, skipping")
            continue
        try:
            entries = loader()
            logger.info(f"Loaded {len(entries)} entries from {ds_name}")
            all_entries.extend(entries)
        except Exception as e:
            logger.warning(f"Error loading {ds_name}: {e}")

    random.shuffle(all_entries)

    if CONFIG.dataset.MAX_VIDEOS_PER_CLASS:
        all_entries = _balance_classes(all_entries, CONFIG.dataset.MAX_VIDEOS_PER_CLASS)

    logger.info(f"Total entries: {len(all_entries)}")
    return all_entries


def _balance_classes(entries, max_per_class):
    """Limit entries per class for balanced training."""
    by_label = {}
    for e in entries:
        by_label.setdefault(e["label"], []).append(e)
    balanced = []
    for label, items in by_label.items():
        random.shuffle(items)
        balanced.extend(items[:max_per_class])
    random.shuffle(balanced)
    return balanced


def split_entries(entries: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split entries into train / val / test."""
    n = len(entries)
    n_train = int(n * CONFIG.dataset.TRAIN_RATIO)
    n_val = int(n * CONFIG.dataset.VAL_RATIO)
    return entries[:n_train], entries[n_train:n_train + n_val], entries[n_train + n_val:]


# ──────────────── PYTORCH DATASET ────────────────────

class TrueFrameReelsDataset(Dataset):
    """
    PyTorch Dataset for TrueFrame Reels deepfake detection.

    Each sample is a sequence of face crops from a video, returned as a
    tensor of shape (SEQUENCE_LENGTH, 3, H, W).
    """

    def __init__(
        self,
        entries: List[Dict[str, Any]],
        transform=None,
        is_train: bool = True,
    ):
        self.entries = entries
        self.transform = transform
        self.is_train = is_train
        self.extractor = ReelsFrameExtractor()

        # Manipulation type → index mapping
        self.manip_to_idx = {
            t: i for i, t in enumerate(CONFIG.model.MANIPULATION_TYPES)
        }

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[idx]

        # Extract face sequence
        face_crops, meta = self.extractor.extract_face_sequence(entry["path"])

        if face_crops is None:
            # Return a zero tensor if extraction fails — collate_fn will filter
            seq_len = CONFIG.frames.SEQUENCE_LENGTH
            h, w = CONFIG.frames.FRAME_SIZE
            return {
                "frames": torch.zeros(seq_len, 3, h, w),
                "label": torch.tensor(entry["label"], dtype=torch.long),
                "manipulation_type": torch.tensor(
                    self.manip_to_idx.get(entry["manipulation_type"], 0),
                    dtype=torch.long,
                ),
                "valid": torch.tensor(0),
            }

        # Apply augmentations to each frame
        processed = []
        for crop in face_crops:
            # BGR → RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            if self.transform:
                augmented = self.transform(image=crop_rgb)
                crop_tensor = augmented["image"]
            else:
                crop_tensor = torch.from_numpy(
                    crop_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
                )
            processed.append(crop_tensor)

        # Stack into (SEQUENCE_LENGTH, 3, H, W)
        frames_tensor = torch.stack(processed)

        return {
            "frames": frames_tensor,
            "label": torch.tensor(entry["label"], dtype=torch.long),
            "manipulation_type": torch.tensor(
                self.manip_to_idx.get(entry["manipulation_type"], 0),
                dtype=torch.long,
            ),
            "valid": torch.tensor(1),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate that filters out invalid samples."""
    valid_batch = [b for b in batch if b["valid"].item() == 1]
    if not valid_batch:
        # Return at least one dummy sample to avoid DataLoader crash
        return batch[0] if batch else {}

    return {
        "frames": torch.stack([b["frames"] for b in valid_batch]),
        "label": torch.stack([b["label"] for b in valid_batch]),
        "manipulation_type": torch.stack([b["manipulation_type"] for b in valid_batch]),
        "valid": torch.stack([b["valid"] for b in valid_batch]),
    }


# ────────────── DATALOADER FACTORY ───────────────────

def create_dataloaders(
    datasets: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders from configured datasets.
    Returns (train_loader, val_loader, test_loader).
    """

    entries = load_all_entries(datasets)
    train_entries, val_entries, test_entries = split_entries(entries)

    logger.info(
        f"Split: train={len(train_entries)}, val={len(val_entries)}, test={len(test_entries)}"
    )

    train_transform = get_train_transforms() if CONFIG.training.USE_AUGMENTATION else get_val_transforms()
    val_transform = get_val_transforms()

    train_ds = TrueFrameReelsDataset(train_entries, train_transform, is_train=True)
    val_ds = TrueFrameReelsDataset(val_entries, val_transform, is_train=False)
    test_ds = TrueFrameReelsDataset(test_entries, val_transform, is_train=False)

    # Weighted sampler for class balance
    sampler = None
    if CONFIG.dataset.OVERSAMPLE_MINORITY:
        labels = [e["label"] for e in train_entries]
        class_counts = np.bincount(labels)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = [class_weights[l] for l in labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG.training.BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=CONFIG.training.NUM_WORKERS,
        pin_memory=CONFIG.training.PIN_MEMORY,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG.training.BATCH_SIZE,
        shuffle=False,
        num_workers=CONFIG.training.NUM_WORKERS,
        pin_memory=CONFIG.training.PIN_MEMORY,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG.training.BATCH_SIZE,
        shuffle=False,
        num_workers=CONFIG.training.NUM_WORKERS,
        pin_memory=CONFIG.training.PIN_MEMORY,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader
