"""
TrueFrame Reels — Evaluation Metrics
======================================
Metric tracking for deepfake detection: AUC-ROC, AUC-PR,
Accuracy, Precision, Recall, F1, EER, per-threshold analysis.
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    roc_curve, confusion_matrix, log_loss,
)


class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.all_probs = []
        self.all_labels = []

    def update(self, probs: np.ndarray, labels: np.ndarray):
        self.all_probs.extend(probs.flatten().tolist())
        self.all_labels.extend(labels.flatten().tolist())

    def compute(self, threshold: float = 0.5) -> Dict[str, float]:
        if not self.all_probs:
            return self._empty()

        probs = np.array(self.all_probs)
        labels = np.array(self.all_labels)
        preds = (probs >= threshold).astype(int)

        m = {
            "accuracy": float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "f1_score": float(f1_score(labels, preds, zero_division=0)),
        }

        try:
            m["auc_roc"] = float(roc_auc_score(labels, probs))
        except ValueError:
            m["auc_roc"] = 0.0

        try:
            m["auc_pr"] = float(average_precision_score(labels, probs))
        except ValueError:
            m["auc_pr"] = 0.0

        try:
            m["log_loss"] = float(log_loss(labels, probs))
        except ValueError:
            m["log_loss"] = float("inf")

        m["eer"] = self._compute_eer(labels, probs)

        cm = confusion_matrix(labels, preds, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            m.update({"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)})
            m["fpr"] = float(fp / (fp + tn + 1e-8))
            m["fnr"] = float(fn / (fn + tp + 1e-8))

        return m

    def _compute_eer(self, labels, probs):
        try:
            fpr, tpr, _ = roc_curve(labels, probs)
            fnr = 1 - tpr
            idx = np.nanargmin(np.abs(fpr - fnr))
            return float((fpr[idx] + fnr[idx]) / 2)
        except Exception:
            return 1.0

    def _empty(self):
        return {k: 0.0 for k in [
            "accuracy", "precision", "recall", "f1_score",
            "auc_roc", "auc_pr", "eer",
        ]}


def threshold_analysis(probs, labels, thresholds=None):
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        results.append({
            "threshold": t,
            "accuracy": float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
        })
    return results


def per_type_evaluation(probs, labels, manip_types, type_names, threshold=0.5):
    results = {}
    for idx, name in enumerate(type_names):
        mask = manip_types == idx
        if not np.any(mask):
            continue
        tp, tl = probs[mask], labels[mask]
        pred = (tp >= threshold).astype(int)
        r = {
            "count": int(np.sum(mask)),
            "accuracy": float(accuracy_score(tl, pred)),
            "f1": float(f1_score(tl, pred, zero_division=0)),
        }
        try:
            r["auc_roc"] = float(roc_auc_score(tl, tp))
        except ValueError:
            r["auc_roc"] = 0.0
        results[name] = r
    return results
