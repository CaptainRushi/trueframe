"""
TrueFrame Reels — EfficientNet-B4 + LSTM Model
=================================================
Spatio-temporal deepfake detection model.

Architecture:
    Frame → EfficientNet-B4 (CNN Feature Extractor)
        → Feature Sequence (T × 1792)
        → Bidirectional LSTM (Temporal Encoder)
        → Temporal Attention Pooling
        → Binary Classifier (Real vs Deepfake)
        → (Optional) Manipulation Type Head

Designed for short-form vertical videos (reels, 5–60s).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Optional

from training.config import CONFIG


# ────────────── TEMPORAL ATTENTION ───────────────────

class TemporalAttention(nn.Module):
    """
    Soft attention over temporal LSTM features.
    Learns to weight frames that contribute most to the detection decision.
    Deepfake artifacts are often visible in only a subset of frames.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim) — attention-weighted summary
        """
        # (batch, seq_len, 1)
        weights = self.attention(lstm_output)
        weights = F.softmax(weights, dim=1)
        # Weighted sum: (batch, hidden_dim)
        context = torch.sum(weights * lstm_output, dim=1)
        return context


# ────────────── MAIN MODEL ───────────────────────────

class TrueFrameReelsDetector(nn.Module):
    """
    EfficientNet-B4 + Bidirectional LSTM with Temporal Attention.

    Input:  (batch, seq_len, 3, 224, 224) — sequence of face crops
    Output: {
        "logits": (batch, 2),                           — Real vs Deepfake
        "deepfake_prob": (batch,),                      — Probability of deepfake
        "manipulation_logits": (batch, num_manip_types), — If multi-task enabled
        "temporal_attention": (batch, seq_len),          — Attention weights
    }
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or CONFIG.model

        # ─── CNN Backbone (EfficientNet-B4) ───
        self.backbone = self._build_backbone()
        self.feature_dim = self.cfg.BACKBONE_FEATURE_DIM

        # ─── Feature projection (optional dimensionality reduction) ───
        self.feature_proj = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        projected_dim = 512

        # ─── LSTM Temporal Encoder ───
        self.lstm = nn.LSTM(
            input_size=projected_dim,
            hidden_size=self.cfg.LSTM_HIDDEN_DIM,
            num_layers=self.cfg.LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=self.cfg.LSTM_DROPOUT if self.cfg.LSTM_NUM_LAYERS > 1 else 0,
            bidirectional=self.cfg.LSTM_BIDIRECTIONAL,
        )

        lstm_output_dim = self.cfg.LSTM_HIDDEN_DIM * (2 if self.cfg.LSTM_BIDIRECTIONAL else 1)

        # ─── Temporal Attention ───
        self.use_attention = self.cfg.USE_TEMPORAL_ATTENTION
        if self.use_attention:
            self.temporal_attention = TemporalAttention(lstm_output_dim)

        # ─── Classification Head (Binary: Real vs Deepfake) ───
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(self.cfg.HEAD_DROPOUT),
            nn.Linear(128, self.cfg.NUM_CLASSES),
        )

        # ─── Manipulation Type Head (Multi-task, optional) ───
        self.use_manip_head = self.cfg.USE_MANIPULATION_TYPE_HEAD
        if self.use_manip_head:
            num_manip = len(self.cfg.MANIPULATION_TYPES)
            self.manipulation_head = nn.Sequential(
                nn.Linear(lstm_output_dim, 128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_manip),
            )

    def _build_backbone(self):
        """Build EfficientNet-B4 backbone, removing the classification head."""
        if self.cfg.BACKBONE == "efficientnet_b4":
            weights = models.EfficientNet_B4_Weights.DEFAULT if self.cfg.BACKBONE_PRETRAINED else None
            backbone = models.efficientnet_b4(weights=weights)
        elif self.cfg.BACKBONE == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if self.cfg.BACKBONE_PRETRAINED else None
            backbone = models.efficientnet_b0(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {self.cfg.BACKBONE}")

        # Remove classifier → use features only
        backbone.classifier = nn.Identity()
        return backbone

    def freeze_backbone(self):
        """Freeze CNN backbone weights for initial training epochs."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze CNN backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract CNN features for a batch of frame sequences.
        Args:
            frames: (batch, seq_len, 3, H, W)
        Returns:
            features: (batch, seq_len, feature_dim)
        """
        batch_size, seq_len, C, H, W = frames.shape

        # Reshape to process all frames through CNN at once
        x = frames.view(batch_size * seq_len, C, H, W)

        # Forward through backbone
        features = self.backbone(x)  # (batch*seq_len, feature_dim)

        # Project features
        features = self.feature_proj(features)  # (batch*seq_len, 512)

        # Reshape back to sequences
        features = features.view(batch_size, seq_len, -1)
        return features

    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        Args:
            frames: (batch, seq_len, 3, H, W)
        Returns:
            Dict with logits, probabilities, and attention weights.
        """
        # 1. Extract spatial features
        features = self.extract_features(frames)  # (B, T, 512)

        # 2. Temporal encoding via LSTM
        lstm_out, (h_n, c_n) = self.lstm(features)  # (B, T, lstm_dim)

        # 3. Temporal pooling
        if self.use_attention:
            temporal_repr = self.temporal_attention(lstm_out)  # (B, lstm_dim)
            # Extract attention weights for interpretability
            attn_weights = self.temporal_attention.attention(lstm_out).squeeze(-1)
            attn_weights = F.softmax(attn_weights, dim=1)
        else:
            # Use last hidden state
            temporal_repr = lstm_out[:, -1, :]
            attn_weights = None

        # 4. Binary classification
        logits = self.classifier(temporal_repr)  # (B, 2)
        probs = F.softmax(logits, dim=1)
        deepfake_prob = probs[:, 1]  # P(deepfake)

        output = {
            "logits": logits,
            "deepfake_prob": deepfake_prob,
        }

        if attn_weights is not None:
            output["temporal_attention"] = attn_weights

        # 5. Manipulation type head (multi-task)
        if self.use_manip_head:
            manip_logits = self.manipulation_head(temporal_repr)
            output["manipulation_logits"] = manip_logits

        return output


# ────────────── LOSS FUNCTIONS ───────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    Reduces loss for well-classified examples, focusing on hard cases.
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
    """
    Combined loss for the TrueFrame model.
    Supports:
      - Binary CE / Focal / Label-smoothing CE
      - Multi-task manipulation type loss
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or CONFIG.training

        # Primary loss
        if self.cfg.LOSS_FN == "focal":
            self.primary_loss = FocalLoss(
                alpha=self.cfg.FOCAL_ALPHA,
                gamma=self.cfg.FOCAL_GAMMA,
            )
        elif self.cfg.LOSS_FN == "label_smoothing_ce":
            self.primary_loss = nn.CrossEntropyLoss(
                label_smoothing=self.cfg.LABEL_SMOOTHING
            )
        else:
            self.primary_loss = nn.CrossEntropyLoss()

        # Manipulation type loss
        self.manip_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        manipulation_types: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Binary classification loss
        loss_binary = self.primary_loss(outputs["logits"], labels)

        total_loss = self.cfg.WEIGHT_BINARY * loss_binary
        loss_dict = {"loss_binary": loss_binary}

        # Manipulation type loss (multi-task)
        if (
            manipulation_types is not None
            and "manipulation_logits" in outputs
            and self.cfg.WEIGHT_MANIPULATION_TYPE > 0
        ):
            loss_manip = self.manip_loss(
                outputs["manipulation_logits"], manipulation_types
            )
            total_loss += self.cfg.WEIGHT_MANIPULATION_TYPE * loss_manip
            loss_dict["loss_manipulation_type"] = loss_manip

        loss_dict["loss_total"] = total_loss
        return loss_dict


# ────────────── MODEL FACTORY ────────────────────────

def build_model(cfg=None) -> TrueFrameReelsDetector:
    """Create a fresh TrueFrameReelsDetector model."""
    return TrueFrameReelsDetector(cfg)


def build_loss(cfg=None) -> TrueFrameLoss:
    """Create the combined loss function."""
    return TrueFrameLoss(cfg)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
