"""Evaluation metrics in plain numpy, so they're easy to test."""

from __future__ import annotations

import numpy as np


def topk_accuracy(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Fraction of rows where the true label is among the k highest scores."""
    # argsort is ascending, so the last k columns are the top k.
    topk = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean(np.any(topk == labels[:, None], axis=1)))


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    """Rows are true classes, columns are predicted classes."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    # np.add.at handles repeated (true, pred) pairs correctly.
    np.add.at(cm, (labels, preds), 1)
    return cm


def macro_f1(cm: np.ndarray) -> float:
    """Unweighted mean of per-class F1 scores."""
    tp = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0)
    actual = cm.sum(axis=1)
    # I treat 0/0 as 0, the same as scikit-learn's zero_division=0.
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    actual = cm.sum(axis=1)
    return np.divide(np.diag(cm), actual, out=np.zeros(len(cm)), where=actual > 0)


def most_confused_pairs(cm: np.ndarray, n: int) -> list[tuple[int, int, int, int]]:
    """The n species pairs mixed up most often, counting both directions.

    Returns (class_a, class_b, a_predicted_as_b, b_predicted_as_a), sorted by
    the total number of mix-ups.
    """
    pairs = []
    k = len(cm)
    for a in range(k):
        for b in range(a + 1, k):
            total = cm[a, b] + cm[b, a]
            if total > 0:
                pairs.append((int(total), a, b, int(cm[a, b]), int(cm[b, a])))
    # Sort by total, then by class ids so ties are stable.
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    return [(a, b, ab, ba) for _, a, b, ab, ba in pairs[:n]]


def threshold_table(confidence: np.ndarray, correct: np.ndarray, thresholds) -> list[dict]:
    """For each threshold: how many predictions pass and how accurate they are."""
    rows = []
    for t in thresholds:
        kept = confidence >= t
        rows.append(
            {
                "threshold": float(t),
                # Coverage is the share of photos that would get a confident answer.
                "coverage": float(kept.mean()),
                "accuracy_above": float(correct[kept].mean()) if kept.any() else float("nan"),
                "accuracy_below": float(correct[~kept].mean()) if (~kept).any() else float("nan"),
            }
        )
    return rows
