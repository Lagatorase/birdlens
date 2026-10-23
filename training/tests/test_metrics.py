import numpy as np
import pytest

from birdlens_train.metrics import (
    confusion_matrix,
    macro_f1,
    most_confused_pairs,
    per_class_accuracy,
    threshold_table,
    topk_accuracy,
)


def test_topk_accuracy():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.5, 0.4, 0.1]])
    labels = np.array([0, 1, 2])
    # Only the first row is right at top-1.
    assert topk_accuracy(probs, labels, 1) == pytest.approx(1 / 3)
    # Row 2 has class 1 second, row 3 has class 2 last.
    assert topk_accuracy(probs, labels, 2) == pytest.approx(2 / 3)
    assert topk_accuracy(probs, labels, 3) == 1.0


def test_confusion_matrix_and_per_class_accuracy():
    cm = confusion_matrix(np.array([0, 0, 1, 2, 2]), np.array([0, 1, 1, 2, 0]), 3)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 1]]
    assert per_class_accuracy(cm).tolist() == [0.5, 1.0, 0.5]


def test_macro_f1_perfect_and_known_value():
    perfect = confusion_matrix(np.array([0, 1, 2]), np.array([0, 1, 2]), 3)
    assert macro_f1(perfect) == 1.0
    cm = confusion_matrix(np.array([0, 0, 1, 2, 2]), np.array([0, 1, 1, 2, 0]), 3)
    # Per class: P=[.5,.5,1], R=[.5,1,.5] so F1=[.5, 2/3, 2/3].
    assert macro_f1(cm) == pytest.approx((0.5 + 2 / 3 + 2 / 3) / 3)


def test_macro_f1_handles_a_class_never_predicted():
    cm = confusion_matrix(np.array([0, 1]), np.array([0, 0]), 2)
    # Class 1 has zero precision and recall, which counts as F1 = 0.
    assert macro_f1(cm) == pytest.approx((2 / 3 + 0) / 2)


def test_most_confused_pairs_counts_both_directions():
    cm = np.zeros((4, 4), dtype=int)
    cm[0, 1], cm[1, 0] = 3, 2  # pair (0, 1): 5 mix-ups
    cm[2, 3] = 4  # pair (2, 3): 4 mix-ups
    cm[0, 3] = 1
    pairs = most_confused_pairs(cm, 2)
    assert pairs == [(0, 1, 3, 2), (2, 3, 4, 0)]


def test_threshold_table():
    confidence = np.array([0.9, 0.8, 0.4, 0.3])
    correct = np.array([True, True, False, True])
    row = threshold_table(confidence, correct, [0.5])[0]
    assert row["coverage"] == 0.5
    assert row["accuracy_above"] == 1.0
    assert row["accuracy_below"] == 0.5
