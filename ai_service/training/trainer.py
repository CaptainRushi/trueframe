"""
TrueFrame Reels — Training Engine
====================================
Full training loop with:
  - Mixed-precision training (AMP)
  - Gradient accumulation
  - Cosine warmup LR scheduling
  - Early stopping
  - Backbone freeze/unfreeze schedule
  - Comprehensive metric logging
  - Checkpoint saving
"""

import os
import sys
import json
import time
import logging
import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    LambdaLR,
)

from training.config import CONFIG, CHECKPOINT_DIR, LOG_DIR
from training.model import build_model, build_loss, count_parameters
from training.dataset import create_dataloaders
from training.metrics import MetricTracker

logger = logging.getLogger("trueframe.trainer")


# ────────────── LR SCHEDULERS ────────────────────────

def _cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs, min_lr):
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr / CONFIG.training.LEARNING_RATE,
                   0.5 * (1.0 + np.cos(np.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def build_optimizer(model):
    """Build optimizer with per-parameter group LRs."""
    # Separate backbone and head parameters
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

    param_groups = [
        {"params": head_params, "lr": CONFIG.training.LEARNING_RATE},
        {"params": backbone_params, "lr": CONFIG.training.LEARNING_RATE * 0.1},
    ]

    if CONFIG.training.OPTIMIZER == "adamw":
        return torch.optim.AdamW(
            param_groups,
            weight_decay=CONFIG.training.WEIGHT_DECAY,
            betas=CONFIG.training.BETAS,
        )
    elif CONFIG.training.OPTIMIZER == "adam":
        return torch.optim.Adam(
            param_groups,
            betas=CONFIG.training.BETAS,
        )
    elif CONFIG.training.OPTIMIZER == "sgd":
        return torch.optim.SGD(
            param_groups,
            momentum=0.9,
            weight_decay=CONFIG.training.WEIGHT_DECAY,
        )
    else:
        raise ValueError(f"Unknown optimizer: {CONFIG.training.OPTIMIZER}")


def build_scheduler(optimizer):
    """Build LR scheduler."""
    if CONFIG.training.SCHEDULER == "cosine_warmup":
        return _cosine_warmup_scheduler(
            optimizer,
            CONFIG.training.WARMUP_EPOCHS,
            CONFIG.training.MAX_EPOCHS,
            CONFIG.training.MIN_LR,
        )
    elif CONFIG.training.SCHEDULER == "step":
        return StepLR(
            optimizer,
            step_size=CONFIG.training.STEP_SIZE,
            gamma=CONFIG.training.GAMMA,
        )
    elif CONFIG.training.SCHEDULER == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )
    else:
        return None


# ────────────── EARLY STOPPING ───────────────────────

class EarlyStopping:
    """Stops training when the monitored metric stops improving."""

    def __init__(self, patience=7, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0

    def __call__(self, metric) -> bool:
        if self.best is None:
            self.best = metric
            return False

        if self.mode == "max":
            improved = metric > self.best
        else:
            improved = metric < self.best

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


# ────────────── TRAINING ENGINE ──────────────────────

class TrueFrameTrainer:
    """
    Main training engine for the TrueFrame Reels Deepfake Detector.

    Features:
      - Mixed precision (AMP)
      - Gradient accumulation
      - Backbone freeze → unfreeze schedule
      - Early stopping
      - Best checkpoint saving
      - Training history logging
    """

    def __init__(
        self,
        model=None,
        loss_fn=None,
        device: str = "auto",
        resume_from: Optional[str] = None,
    ):
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Model
        self.model = model or build_model()
        self.model = self.model.to(self.device)
        params = count_parameters(self.model)
        logger.info(
            f"Model parameters: {params['total']:,} total, "
            f"{params['trainable']:,} trainable, {params['frozen']:,} frozen"
        )

        # Loss
        self.loss_fn = loss_fn or build_loss()

        # Optimizer & Scheduler
        self.optimizer = build_optimizer(self.model)
        self.scheduler = build_scheduler(self.optimizer)

        # AMP
        self.use_amp = CONFIG.training.USE_AMP and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=CONFIG.training.EARLY_STOPPING_PATIENCE,
            mode=CONFIG.training.EARLY_STOPPING_MODE,
        )

        # Metrics
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()

        # Training state
        self.current_epoch = 0
        self.best_metric = None
        self.history = []

        # Resume from checkpoint
        if resume_from:
            self._load_checkpoint(resume_from)

    def train(
        self,
        train_loader,
        val_loader,
        max_epochs: Optional[int] = None,
    ) -> Dict:
        """
        Full training loop.
        Returns training history as a dict.
        """
        max_epochs = max_epochs or CONFIG.training.MAX_EPOCHS
        freeze_epochs = CONFIG.model.FREEZE_BACKBONE_EPOCHS
        accumulate = CONFIG.training.ACCUMULATE_GRAD_BATCHES

        # Initial backbone freeze
        if freeze_epochs > 0 and self.current_epoch < freeze_epochs:
            self.model.freeze_backbone()
            logger.info(f"Backbone frozen for first {freeze_epochs} epochs")

        logger.info(f"Starting training for {max_epochs} epochs")
        logger.info(f"Gradient accumulation: {accumulate} steps")

        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch

            # Unfreeze backbone after warmup
            if epoch == freeze_epochs and freeze_epochs > 0:
                self.model.unfreeze_backbone()
                logger.info(f"Epoch {epoch}: Backbone unfrozen for fine-tuning")
                params = count_parameters(self.model)
                logger.info(f"Trainable params now: {params['trainable']:,}")

            # Train one epoch
            train_result = self._train_epoch(train_loader, epoch, accumulate)

            # Validate
            val_result = self._validate_epoch(val_loader, epoch)

            # LR scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_result["val_auc"])
                else:
                    self.scheduler.step()

            # Log
            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_result = {
                "epoch": epoch,
                "lr": current_lr,
                **train_result,
                **val_result,
            }
            self.history.append(epoch_result)
            self._log_epoch(epoch, epoch_result)

            # Save checkpoint
            metric_val = val_result.get(CONFIG.training.CHECKPOINT_METRIC, 0)
            self._save_checkpoint(epoch, metric_val)

            # Early stopping
            es_metric = val_result.get(CONFIG.training.EARLY_STOPPING_METRIC, 0)
            if self.early_stopping(es_metric):
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best {CONFIG.training.EARLY_STOPPING_METRIC}: "
                    f"{self.early_stopping.best:.4f}"
                )
                break

        # Save final training history
        self._save_history()
        return {"history": self.history, "best_metric": self.early_stopping.best}

    def _train_epoch(self, loader, epoch, accumulate) -> Dict:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        self.optimizer.zero_grad()

        total_loss = 0
        num_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(loader):
            if not isinstance(batch, dict) or "frames" not in batch:
                continue

            frames = batch["frames"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            manip_types = batch.get("manipulation_type")
            if manip_types is not None:
                manip_types = manip_types.to(self.device, non_blocking=True)

            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                outputs = self.model(frames)
                losses = self.loss_fn(outputs, labels, manip_types)
                loss = losses["loss_total"] / accumulate

            # Backward with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient accumulation step
            if (batch_idx + 1) % accumulate == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    CONFIG.training.GRADIENT_CLIP_VAL,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Track metrics
            total_loss += losses["loss_total"].item()
            num_batches += 1

            probs = outputs["deepfake_prob"].detach().cpu().numpy()
            gt = labels.cpu().numpy()
            self.train_metrics.update(probs, gt)

        elapsed = time.time() - start_time
        avg_loss = total_loss / max(1, num_batches)
        metrics = self.train_metrics.compute()

        return {
            "train_loss": avg_loss,
            "train_acc": metrics["accuracy"],
            "train_auc": metrics["auc_roc"],
            "train_f1": metrics["f1_score"],
            "train_time": elapsed,
        }

    @torch.no_grad()
    def _validate_epoch(self, loader, epoch) -> Dict:
        """Validate for one epoch."""
        self.model.eval()
        self.val_metrics.reset()

        total_loss = 0
        num_batches = 0

        for batch in loader:
            if not isinstance(batch, dict) or "frames" not in batch:
                continue

            frames = batch["frames"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            manip_types = batch.get("manipulation_type")
            if manip_types is not None:
                manip_types = manip_types.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(frames)
                losses = self.loss_fn(outputs, labels, manip_types)

            total_loss += losses["loss_total"].item()
            num_batches += 1

            probs = outputs["deepfake_prob"].detach().cpu().numpy()
            gt = labels.cpu().numpy()
            self.val_metrics.update(probs, gt)

        avg_loss = total_loss / max(1, num_batches)
        metrics = self.val_metrics.compute()

        return {
            "val_loss": avg_loss,
            "val_acc": metrics["accuracy"],
            "val_auc": metrics["auc_roc"],
            "val_f1": metrics["f1_score"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_eer": metrics["eer"],
        }

    def _save_checkpoint(self, epoch, metric_val):
        """Save model checkpoint if it's the best so far."""
        is_best = False
        if self.best_metric is None:
            is_best = True
        elif CONFIG.training.CHECKPOINT_MODE == "max":
            is_best = metric_val > self.best_metric
        else:
            is_best = metric_val < self.best_metric

        if is_best:
            self.best_metric = metric_val
            path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "best_metric": self.best_metric,
                "config": {
                    "backbone": CONFIG.model.BACKBONE,
                    "lstm_hidden": CONFIG.model.LSTM_HIDDEN_DIM,
                    "lstm_layers": CONFIG.model.LSTM_NUM_LAYERS,
                    "seq_length": CONFIG.frames.SEQUENCE_LENGTH,
                },
            }, path)
            logger.info(
                f"  ✓ Best model saved (epoch {epoch}, "
                f"{CONFIG.training.CHECKPOINT_METRIC}={metric_val:.4f})"
            )

        # Also save periodic checkpoints
        if (epoch + 1) % 5 == 0:
            path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }, path)

    def _load_checkpoint(self, path):
        """Resume training from a checkpoint."""
        logger.info(f"Resuming from checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") and self.scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.current_epoch = ckpt.get("epoch", 0) + 1
        self.best_metric = ckpt.get("best_metric")
        logger.info(f"Resumed at epoch {self.current_epoch}")

    def _log_epoch(self, epoch, result):
        """Log epoch results."""
        logger.info(
            f"Epoch {epoch:3d} | "
            f"LR {result['lr']:.2e} | "
            f"Train Loss {result['train_loss']:.4f} Acc {result['train_acc']:.4f} "
            f"AUC {result['train_auc']:.4f} | "
            f"Val Loss {result['val_loss']:.4f} Acc {result['val_acc']:.4f} "
            f"AUC {result['val_auc']:.4f} F1 {result['val_f1']:.4f} | "
            f"Time {result['train_time']:.1f}s"
        )

    def _save_history(self):
        """Save training history to JSON."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOG_DIR, f"training_history_{timestamp}.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to {path}")


# ────────────── MAIN ENTRY POINT ─────────────────────

def run_training(
    datasets=None,
    resume_from=None,
    device="auto",
    max_epochs=None,
):
    """
    Main training entry point.

    Args:
        datasets: List of dataset names to use (default: all configured)
        resume_from: Path to checkpoint to resume from
        device: "auto", "cuda", "cpu"
        max_epochs: Override max epochs

    Returns:
        Training result dict
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(LOG_DIR, "training.log"),
                mode="a",
            ),
        ],
    )

    logger.info("=" * 70)
    logger.info("TrueFrame Reels — Deepfake Detection Model Training")
    logger.info("=" * 70)
    logger.info(f"Config: {CONFIG.model.BACKBONE} + LSTM")
    logger.info(f"Sequence length: {CONFIG.frames.SEQUENCE_LENGTH}")
    logger.info(f"Frame size: {CONFIG.frames.FRAME_SIZE}")

    # Set random seeds
    torch.manual_seed(CONFIG.training.SEED)
    np.random.seed(CONFIG.training.SEED)

    # Create data loaders
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(datasets)

    # Build model & trainer
    trainer = TrueFrameTrainer(
        device=device,
        resume_from=resume_from,
    )

    # Train
    result = trainer.train(train_loader, val_loader, max_epochs)

    # Final test evaluation
    logger.info("Running final evaluation on test set...")
    test_result = trainer._validate_epoch(test_loader, epoch=-1)
    logger.info(
        f"Test Results: "
        f"Loss {test_result['val_loss']:.4f} | "
        f"Acc {test_result['val_acc']:.4f} | "
        f"AUC {test_result['val_auc']:.4f} | "
        f"F1 {test_result['val_f1']:.4f} | "
        f"EER {test_result['val_eer']:.4f}"
    )

    result["test"] = test_result
    return result


# ────────────── CLI ──────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train TrueFrame Reels Deepfake Detector")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset names to use (default: all)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume training from")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")

    args = parser.parse_args()

    # Apply CLI overrides
    if args.batch_size:
        CONFIG.training.BATCH_SIZE = args.batch_size
    if args.lr:
        CONFIG.training.LEARNING_RATE = args.lr

    result = run_training(
        datasets=args.datasets,
        resume_from=args.resume,
        device=args.device,
        max_epochs=args.epochs,
    )

    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Best validation AUC: {result.get('best_metric', 'N/A')}")
    print("=" * 70)
