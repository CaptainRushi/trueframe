"""
TrueFrame Reels — Data Augmentation Pipeline
==============================================
Albumentations-based augmentation pipeline optimized for deepfake detection
on short-form social media videos (reels).

Key considerations:
  - Compression artifact simulation (JPEG quality variance)
  - Mobile camera noise patterns
  - Social media re-encoding effects
  - Face-specific augmentations
"""

import cv2
import numpy as np

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALBUMENTATIONS = True
except ImportError:
    _HAS_ALBUMENTATIONS = False

from training.config import CONFIG


def get_train_transforms():
    """
    Full training augmentation pipeline.
    Simulates real-world conditions: compression, blur, noise,
    lighting changes, and geometric distortions.
    """
    cfg = CONFIG.augmentation
    size = CONFIG.frames.FRAME_SIZE

    if not _HAS_ALBUMENTATIONS:
        return _get_basic_transform(size, cfg)

    return A.Compose([
        # ──── Geometric ────
        A.HorizontalFlip(p=cfg.RANDOM_HORIZONTAL_FLIP),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=(cfg.RANDOM_SCALE[0] - 1, cfg.RANDOM_SCALE[1] - 1),
            rotate_limit=cfg.RANDOM_ROTATION_DEGREES,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
        A.RandomResizedCrop(
            size=size,
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            p=0.3,
        ),
        A.Resize(*size),

        # ──── Photometric ────
        A.OneOf([
            A.ColorJitter(
                brightness=cfg.COLOR_JITTER_BRIGHTNESS,
                contrast=cfg.COLOR_JITTER_CONTRAST,
                saturation=cfg.COLOR_JITTER_SATURATION,
                hue=cfg.COLOR_JITTER_HUE,
                p=1.0,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0,
            ),
        ], p=0.6),

        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=15,
            p=0.3,
        ),

        # ──── Compression + Noise (Social Media Simulation) ────
        A.OneOf([
            A.ImageCompression(
                quality_lower=cfg.JPEG_QUALITY_RANGE[0],
                quality_upper=cfg.JPEG_QUALITY_RANGE[1],
                p=1.0,
            ),
            A.Downscale(
                scale_min=0.5,
                scale_max=0.9,
                p=1.0,
            ),
        ], p=cfg.JPEG_COMPRESSION_PROB),

        A.GaussianBlur(
            blur_limit=cfg.GAUSSIAN_BLUR_KERNEL,
            p=cfg.GAUSSIAN_BLUR_PROB,
        ),

        A.GaussNoise(
            std_range=(cfg.GAUSSIAN_NOISE_STD * 0.5 * 255, cfg.GAUSSIAN_NOISE_STD * 255),
            p=cfg.GAUSSIAN_NOISE_PROB,
        ),

        # ──── Advanced Noise ────
        A.OneOf([
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=1.0),
        ], p=0.15),

        # ──── Occlusion (simulates reels with text overlays, stickers) ────
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(int(size[0] * 0.05), int(size[0] * 0.15)),
            hole_width_range=(int(size[1] * 0.05), int(size[1] * 0.15)),
            fill="random",
            p=0.1,
        ),

        # ──── Normalize + ToTensor ────
        A.Normalize(
            mean=list(cfg.NORMALIZE_MEAN),
            std=list(cfg.NORMALIZE_STD),
        ),
        ToTensorV2(),
    ])


def get_val_transforms():
    """Validation / test transforms — minimal processing."""
    cfg = CONFIG.augmentation
    size = CONFIG.frames.FRAME_SIZE

    if not _HAS_ALBUMENTATIONS:
        return _get_basic_transform(size, cfg)

    return A.Compose([
        A.Resize(*size),
        A.Normalize(
            mean=list(cfg.NORMALIZE_MEAN),
            std=list(cfg.NORMALIZE_STD),
        ),
        ToTensorV2(),
    ])


class _BasicTransform:
    """Fallback transform when albumentations is not installed."""

    def __init__(self, size, mean, std):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image):
        import torch
        img = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # HWC → CHW
        return {"image": torch.from_numpy(img)}


def _get_basic_transform(size, cfg):
    """Fallback when albumentations is not available."""
    return _BasicTransform(size, cfg.NORMALIZE_MEAN, cfg.NORMALIZE_STD)
