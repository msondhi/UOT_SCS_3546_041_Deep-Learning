"""Evaluation metrics. Positive class = fake (label 1)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(labels, probs, threshold: float = 0.5) -> dict[str, float]:
    """Standard classification metrics plus the two identity-verification error rates.

    FAR (False Acceptance Rate): share of fake faces the model lets through as real.
    FRR (False Rejection Rate): share of real customers the model wrongly flags as fake.
    """
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    both_classes = len(np.unique(labels)) == 2
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)) if both_classes else float("nan"),
        "far": float(fn / (tp + fn)) if tp + fn else float("nan"),
        "frr": float(fp / (fp + tn)) if fp + tn else float("nan"),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
