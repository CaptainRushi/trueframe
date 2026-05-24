"""
TrueFrame Reels — Training Configuration
==========================================
Central configuration for the deepfake detection training pipeline.
Covers dataset paths, model hyperparameters, training schedule,
and deployment settings.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

# ─────────────────────── PATHS ───────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
AI_SERVICE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Default directories (override via environment or CLI)
DATA_DIR = os.environ.get("TRUEFRAME_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
CHECKPOINT_DIR = os.environ.get("TRUEFRAME_CKPT_DIR", os.path.join(BASE_DIR, "checkpoints"))
EXPORT_DIR = os.environ.get("TRUEFRAME_EXPORT_DIR", os.path.join(AI_SERVICE_DIR, "models"))
LOG_DIR = os.environ.get("TRUEFRAME_LOG_DIR", os.path.join(BASE_DIR, "logs"))

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ──────────────────── DATASET CONFIG ─────────────────

@dataclass
class DatasetConfig:
    """Configuration for dataset ingestion and preprocessing."""

    # Supported dataset formats / sources
    SUPPORTED_DATASETS: List[str] = field(default_factory=lambda: [
        "faceforensics",       # FaceForensics++ (FF++)
        "dfdc",                # DeepFake Detection Challenge
        "celeb_df",            # Celeb-DF v2
        "custom",              # Custom video folder
    ])

    # Root paths per dataset (set these to your local dataset locations)
    FACEFORENSICS_ROOT: str = os.path.join(DATA_DIR, "FaceForensics")
    DFDC_ROOT: str = os.path.join(DATA_DIR, "DFDC")
    CELEB_DF_ROOT: str = os.path.join(DATA_DIR, "CelebDF")
    CUSTOM_ROOT: str = os.path.join(DATA_DIR, "custom_reels")

    # Split ratios
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # Class balance
    OVERSAMPLE_MINORITY: bool = True
    MAX_VIDEOS_PER_CLASS: Optional[int] = None  # None = no limit


# ──────────────── FRAME EXTRACTION CONFIG ────────────

@dataclass
class FrameExtractionConfig:
    """Settings for extracting frames from reel videos."""

    # Frames per second to sample (1 frame every 0.5s = 2 FPS)
    SAMPLE_FPS: float = 2.0

    # Maximum frames per video clip (for memory efficiency)
    MAX_FRAMES_PER_VIDEO: int = 32

    # Sequence length fed to GRU (key frames selected after SSIM dedup)
    SEQUENCE_LENGTH: int = 20

    # Frame sampling strategy: "uniform", "keyframe", "beginning_middle_end"
    SAMPLING_STRATEGY: str = "uniform"

    # Target frame size after crop+resize (face crop → this size)
    FRAME_SIZE: tuple = (224, 224)

    # Video constraints (TrueFrame reels spec)
    MIN_DURATION_SEC: float = 5.0
    MAX_DURATION_SEC: float = 60.0

    # Face detection minimum confidence
    FACE_CONFIDENCE_THRESHOLD: float = 0.5

    # Minimum face area ratio (face pixels / frame pixels)
    MIN_FACE_AREA_RATIO: float = 0.02


# ──────────────── MODEL ARCHITECTURE CONFIG ──────────

@dataclass
class ModelConfig:
    """
    LightFakeDetect: MobileNetV2 + CBAM + GRU configuration.
    Based on the MDPI LightFakeDetect paper architecture.
    """

    # CNN Backbone (MobileNetV2 — lightweight and fast)
    BACKBONE: str = "mobilenet_v2"             # MobileNetV2 (was: efficientnet_b4)
    BACKBONE_PRETRAINED: bool = True           # Use ImageNet pretrained weights
    BACKBONE_FEATURE_DIM: int = 1280           # MobileNetV2 output dim (was: 1792)
    FREEZE_BACKBONE_EPOCHS: int = 3            # Freeze CNN for N epochs, then fine-tune

    # CBAM Attention Module
    USE_CBAM: bool = True                      # Channel + Spatial attention
    CBAM_REDUCTION_RATIO: int = 16             # Channel reduction ratio in CBAM
    CBAM_SPATIAL_KERNEL: int = 7               # Spatial attention conv kernel size

    # GRU Temporal Encoder (was: BiLSTM)
    GRU_HIDDEN_DIM: int = 256                  # GRU hidden state dimension
    GRU_NUM_LAYERS: int = 2                    # Number of stacked GRU layers
    GRU_DROPOUT: float = 0.3                   # Dropout between GRU layers

    # Keep for backwards compatibility
    LSTM_HIDDEN_DIM: int = 256
    LSTM_NUM_LAYERS: int = 2
    LSTM_DROPOUT: float = 0.3
    LSTM_BIDIRECTIONAL: bool = False           # GRU is unidirectional

    # Classification Head
    NUM_CLASSES: int = 2                       # [Real, Deepfake]
    HEAD_DROPOUT: float = 0.3                  # Dropout before final FC layer

    # Multi-task (kept for schema compatibility, unused in LightFakeDetect)
    USE_MANIPULATION_TYPE_HEAD: bool = False
    MANIPULATION_TYPES: List[str] = field(default_factory=lambda: [
        "real",
        "face_swap",
        "face_reenactment",
        "lip_sync",
        "ai_generated",
    ])

    # Temporal attention (not used in LightFakeDetect — GRU last state is used)
    USE_TEMPORAL_ATTENTION: bool = False


# ──────────────── TRAINING HYPERPARAMETERS ───────────

@dataclass
class TrainingConfig:
    """Training schedule and optimization settings."""

    # Basics
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    MAX_EPOCHS: int = 40
    SEED: int = 42

    # Optimizer
    OPTIMIZER: str = "adamw"                   # "adam", "adamw", "sgd"
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-2
    BETAS: tuple = (0.9, 0.999)

    # LR Schedule
    SCHEDULER: str = "cosine_warmup"           # "cosine_warmup", "step", "plateau"
    WARMUP_EPOCHS: int = 3
    MIN_LR: float = 1e-6
    STEP_SIZE: int = 10                        # For StepLR
    GAMMA: float = 0.1                         # For StepLR

    # Loss
    LOSS_FN: str = "bce_with_logits"           # "bce_with_logits", "focal", "label_smoothing_ce"
    FOCAL_ALPHA: float = 0.25
    FOCAL_GAMMA: float = 2.0
    LABEL_SMOOTHING: float = 0.1

    # Multi-task loss weights
    WEIGHT_BINARY: float = 1.0
    WEIGHT_MANIPULATION_TYPE: float = 0.3

    # Regularization
    MIXUP_ALPHA: float = 0.2
    CUTMIX_ALPHA: float = 1.0
    USE_AUGMENTATION: bool = True

    # Gradient Management
    GRADIENT_CLIP_VAL: float = 1.0
    ACCUMULATE_GRAD_BATCHES: int = 4

    # Early Stopping
    EARLY_STOPPING_PATIENCE: int = 7
    EARLY_STOPPING_METRIC: str = "val_auc"
    EARLY_STOPPING_MODE: str = "max"

    # Checkpointing
    SAVE_TOP_K: int = 3
    CHECKPOINT_METRIC: str = "val_auc"
    CHECKPOINT_MODE: str = "max"

    # Mixed Precision
    USE_AMP: bool = True


# ────────────── AUGMENTATION CONFIG ──────────────────

@dataclass
class AugmentationConfig:
    """Data augmentation settings for training robustness."""

    # Spatial
    RANDOM_HORIZONTAL_FLIP: float = 0.5
    RANDOM_ROTATION_DEGREES: int = 15
    RANDOM_SCALE: tuple = (0.85, 1.15)
    RANDOM_CROP_PAD: int = 16

    # Color / Photometric
    COLOR_JITTER_BRIGHTNESS: float = 0.3
    COLOR_JITTER_CONTRAST: float = 0.3
    COLOR_JITTER_SATURATION: float = 0.2
    COLOR_JITTER_HUE: float = 0.1

    # Compression simulation (critical for social media reels)
    JPEG_QUALITY_RANGE: tuple = (30, 95)
    JPEG_COMPRESSION_PROB: float = 0.4

    # Blur / Noise
    GAUSSIAN_BLUR_PROB: float = 0.2
    GAUSSIAN_BLUR_KERNEL: tuple = (3, 7)
    GAUSSIAN_NOISE_PROB: float = 0.15
    GAUSSIAN_NOISE_STD: float = 0.02

    # Video-specific
    RANDOM_FRAME_DROP_PROB: float = 0.1
    RANDOM_TEMPORAL_SHUFFLE_PROB: float = 0.05

    # Normalization (ImageNet)
    NORMALIZE_MEAN: tuple = (0.485, 0.456, 0.406)
    NORMALIZE_STD: tuple = (0.229, 0.224, 0.225)


# ────────────── DEPLOYMENT / EXPORT CONFIG ───────────

@dataclass
class DeploymentConfig:
    """Settings for model export and real-time inference."""

    # ONNX Export
    EXPORT_ONNX: bool = True
    ONNX_OPSET_VERSION: int = 17
    ONNX_DYNAMIC_AXES: bool = True                   # Dynamic batch size

    # TorchScript Export
    EXPORT_TORCHSCRIPT: bool = True

    # Inference constraints
    MAX_INFERENCE_FRAMES: int = 15                    # Frames per reel at inference
    TARGET_INFERENCE_TIME_MS: int = 2000              # Under 2 seconds total
    USE_GPU_INFERENCE: bool = True

    # Decision thresholds (aligned with existing TrueFrame system)
    THRESHOLD_REAL: float = 0.40                      # < 0.40 → APPROVED
    THRESHOLD_REVIEW: float = 0.60                    # 0.40-0.60 → UNDER_REVIEW
    THRESHOLD_DEEPFAKE: float = 0.80                  # >= 0.80 → REJECTED

    # Score mapping
    SCORE_TABLE = {
        "real": (0.0, 0.40),
        "under_review": (0.40, 0.60),
        "suspected": (0.60, 0.80),
        "deepfake": (0.80, 1.0),
    }


# ────────────── EVALUATION METRICS CONFIG ────────────

@dataclass
class EvaluationConfig:
    """Metrics and evaluation settings."""

    METRICS: List[str] = field(default_factory=lambda: [
        "accuracy",
        "auc_roc",
        "auc_pr",
        "f1_score",
        "precision",
        "recall",
        "eer",                  # Equal Error Rate
        "log_loss",
    ])

    # Per-manipulation-type evaluation
    EVALUATE_PER_TYPE: bool = True

    # Robustness evaluation
    EVALUATE_COMPRESSION_LEVELS: List[int] = field(default_factory=lambda: [
        95, 80, 60, 40, 30
    ])

    # Video-level aggregation strategies
    VIDEO_AGGREGATION: str = "mean"   # "mean", "max", "voting"


# ────────────── GLOBAL CONFIG SINGLETON ──────────────

class TrueFrameTrainingConfig:
    """Master config combining all sub-configs."""

    def __init__(self):
        self.dataset = DatasetConfig()
        self.frames = FrameExtractionConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.augmentation = AugmentationConfig()
        self.deployment = DeploymentConfig()
        self.evaluation = EvaluationConfig()

    def __repr__(self):
        sections = [
            f"Dataset:       {self.dataset}",
            f"Frames:        {self.frames}",
            f"Model:         {self.model}",
            f"Training:      {self.training}",
            f"Augmentation:  {self.augmentation}",
            f"Deployment:    {self.deployment}",
            f"Evaluation:    {self.evaluation}",
        ]
        return "\n".join(sections)


# Default global config instance
CONFIG = TrueFrameTrainingConfig()
