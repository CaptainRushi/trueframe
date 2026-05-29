"""
TrueFrame — LightFakeDetect Model
====================================
Architecture: MobileNetV2 → CBAM → GRU → Binary Classifier
Based on: "LightFakeDetect: A Lightweight Deepfake Video Detection Architecture"
          (MDPI, 2024)

Pipeline per video:
  Frame sequence (T × 3 × 224 × 224)
    → MobileNetV2 backbone          (spatial features, 1280-dim)
    → CBAM attention module         (channel + spatial attention)
    → GRU temporal encoder          (sequence modeling, hidden=256)
    → Classifier head               (FC → ReLU → Dropout → FC → Softmax)
    → P(fake) → binary verdict

Why MobileNet + GRU instead of EfficientNet + LSTM:
  - MobileNet is 4× lighter → faster CPU inference
  - GRU has fewer parameters than LSTM (~33% fewer) → faster convergence
  - CBAM focuses on manipulation artifact regions automatically
  - Same detection accuracy at lower compute cost (MDPI paper benchmarks)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Optional

from training.config import CONFIG


# ─────────────────── CBAM ────────────────────────────

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (from CBAM, Woo et al. 2018).
    Learns which feature channels (filters) are important for detection.
    GAN artifacts often cluster in specific frequency-sensitive channels.

    Uses both global average-pool and global max-pool paths for robustness.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        mid = max(1, in_channels // reduction_ratio)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            attention-scaled x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        # Average pool path
        avg = x.view(B, C, -1).mean(dim=2)           # (B, C)
        avg_out = self.shared_mlp(avg)                 # (B, C)
        # Max pool path
        mx = x.view(B, C, -1).max(dim=2).values      # (B, C)
        mx_out = self.shared_mlp(mx)                   # (B, C)
        # Combine → sigmoid → reshape → scale
        attn = torch.sigmoid(avg_out + mx_out)         # (B, C)
        attn = attn.view(B, C, 1, 1)                  # (B, C, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (from CBAM).
    Learns WHERE in the image the important regions are.
    For deepfakes, this focuses on face-paste seams, eye regions, and
    skin-tone boundaries — exactly where GAN artifacts appear.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            attention-scaled x: (B, C, H, W)
        """
        # Channel-wise avg and max → concat along channel dim
        avg_pool = x.mean(dim=1, keepdim=True)         # (B, 1, H, W)
        max_pool = x.max(dim=1, keepdim=True).values   # (B, 1, H, W)
        spatial  = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, H, W)
        attn = torch.sigmoid(self.bn(self.conv(spatial)))   # (B, 1, H, W)
        return x * attn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Sequential channel attention → spatial attention.

    Adds minimal overhead (~small params) while substantially improving
    the model's ability to focus on manipulation artifact regions.
    Compatible with any CNN backbone as a plug-in module.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn  = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attn  = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ─────────────────── BACKBONE ────────────────────────

class MobileNetV2Backbone(nn.Module):
    """
    MobileNetV2 feature extractor.
    Removes the classification head; outputs 1280-dim feature vector
    after global average pooling.

    Pretrained on ImageNet — leverages rich visual representations
    before fine-tuning for deepfake detection.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        mobilenet = models.mobilenet_v2(weights=weights)

        # Keep only feature layers (remove final classifier)
        self.features = mobilenet.features   # outputs (B, 1280, 7, 7) for 224×224 input

        # Inject CBAM after the backbone's final feature map
        self.cbam = CBAM(in_channels=1280, reduction_ratio=16, spatial_kernel=7)

        # Global average pool → 1280-dim vector
        self.pool = nn.AdaptiveAvgPool2d(1)

    @property
    def output_dim(self) -> int:
        return 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) — normalized face crops
        Returns:
            features: (B, 1280)
        """
        feat = self.features(x)      # (B, 1280, 7, 7)
        feat = self.cbam(feat)       # (B, 1280, 7, 7) — attention applied
        feat = self.pool(feat)       # (B, 1280, 1, 1)
        feat = feat.flatten(1)       # (B, 1280)
        return feat

    def freeze(self):
        """Freeze MobileNet weights for initial training epochs."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze MobileNet for fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True


# ─────────────────── MAIN MODEL ──────────────────────

class LightFakeDetect(nn.Module):
    """
    LightFakeDetect: MobileNetV2 + CBAM + GRU deepfake detector.

    Input:  (batch, seq_len, 3, 224, 224) — normalized face crop sequence
    Output: {
        "logits":         (batch, 2),    — [P(real), P(fake)]
        "deepfake_prob":  (batch,),      — scalar probability of being fake
    }

    Key design decisions:
    - Unidirectional GRU (not BiLSTM) → causal, supports streaming inference
    - GRU over LSTM: ~33% fewer params, similar accuracy, faster training
    - CBAM before temporal modeling: forces backbone to extract manipulation-aware features
    - Freeze backbone for first N epochs → let CBAM + GRU learn before fine-tuning CNN
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or CONFIG.model

        # ─── Backbone (MobileNetV2 + CBAM) ────────────
        self.backbone = MobileNetV2Backbone(pretrained=self.cfg.BACKBONE_PRETRAINED)
        feat_dim = self.backbone.output_dim  # 1280

        # ─── Feature projection (reduce before GRU) ───
        # 1280 → 256 to match GRU hidden size (avoids bottleneck)
        self.feature_proj = nn.Sequential(
            nn.Linear(feat_dim, self.cfg.GRU_HIDDEN_DIM),
            nn.LayerNorm(self.cfg.GRU_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # ─── GRU Temporal Encoder ─────────────────────
        self.gru = nn.GRU(
            input_size=self.cfg.GRU_HIDDEN_DIM,
            hidden_size=self.cfg.GRU_HIDDEN_DIM,
            num_layers=self.cfg.GRU_NUM_LAYERS,
            batch_first=True,
            dropout=self.cfg.GRU_DROPOUT if self.cfg.GRU_NUM_LAYERS > 1 else 0.0,
            bidirectional=False,   # Unidirectional → causal + lighter
        )

        gru_out_dim = self.cfg.GRU_HIDDEN_DIM  # 256

        # ─── Binary Classifier Head ────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.HEAD_DROPOUT),
            nn.Linear(64, self.cfg.NUM_CLASSES),   # 2 classes: real / fake
        )

    def freeze_backbone(self):
        """Freeze MobileNet weights (CBAM + GRU + head remain trainable)."""
        self.backbone.freeze()

    def unfreeze_backbone(self):
        """Unfreeze all weights for fine-tuning."""
        self.backbone.unfreeze()

    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Run MobileNet + CBAM on each frame in the sequence.

        Args:
            frames: (batch, seq_len, 3, H, W)
        Returns:
            features: (batch, seq_len, GRU_HIDDEN_DIM)
        """
        B, T, C, H, W = frames.shape
        # Flatten batch × time → process all frames through CNN at once
        x = frames.view(B * T, C, H, W)          # (B*T, 3, 224, 224)
        feats = self.backbone(x)                   # (B*T, 1280)
        feats = self.feature_proj(feats)           # (B*T, 256)
        feats = feats.view(B, T, -1)              # (B, T, 256)
        return feats

    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            frames: (batch, seq_len, 3, 224, 224)
        Returns:
            dict with logits and deepfake_prob
        """
        # 1. Extract spatial features (MobileNet + CBAM)
        features = self.extract_features(frames)   # (B, T, 256)

        # 2. Temporal modeling (GRU)
        gru_out, h_n = self.gru(features)         # gru_out: (B, T, 256)

        # 3. Use last GRU hidden state as sequence representation
        temporal_repr = gru_out[:, -1, :]         # (B, 256)

        # 4. Binary classification
        logits = self.classifier(temporal_repr)    # (B, 2)
        probs  = F.softmax(logits, dim=1)
        deepfake_prob = probs[:, 1]               # P(fake)

        return {
            "logits":        logits,
            "deepfake_prob": deepfake_prob,
        }


# ─────────────────── LOSS FUNCTIONS ──────────────────

class FocalLoss(nn.Module):
    """
    Focal loss for class-imbalanced deepfake datasets.
    Celeb-DF training set: 711 real / 4511 fake (≈ 1:6.3 ratio).
    Focal loss down-weights easy well-classified examples,
    forcing the model to focus on hard ambiguous cases.

    alpha: weight for the minority (real) class
    gamma: focusing parameter (2.0 is standard)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class TrueFrameLoss(nn.Module):
    """Combined loss function for LightFakeDetect training."""

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or CONFIG.training
        if self.cfg.LOSS_FN == "focal":
            self.loss_fn = FocalLoss(
                alpha=self.cfg.FOCAL_ALPHA,
                gamma=self.cfg.FOCAL_GAMMA,
            )
        elif self.cfg.LOSS_FN == "label_smoothing_ce":
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.cfg.LABEL_SMOOTHING)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        manipulation_types: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        loss = self.loss_fn(outputs["logits"], labels)
        return {"loss_binary": loss, "loss_total": loss}


# ─────────────────── MODEL FACTORY ───────────────────

def build_model(cfg=None) -> LightFakeDetect:
    """Instantiate a LightFakeDetect model."""
    return LightFakeDetect(cfg)


def build_loss(cfg=None) -> TrueFrameLoss:
    """Instantiate the combined loss function."""
    return TrueFrameLoss(cfg)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
