"""Reading the CUB-200-2011 metadata and making the train, val and test splits.

This module only uses the standard library and numpy, so the tests can run
without TensorFlow.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .labels import clean_label


@dataclass(frozen=True)
class Sample:
    image_id: int
    path: str  # relative to the images/ folder
    label: int  # 0-based class index
    is_train: bool


def _read_pairs(path: Path) -> dict[int, str]:
    # Every CUB metadata file is "<id> <value>" on each line.
    pairs: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(" ", 1)
            pairs[int(key)] = value
    return pairs


def load_class_names(root: Path) -> list[str]:
    """Return the raw class folder names in class index order."""
    classes = _read_pairs(root / "classes.txt")
    # CUB class ids start at 1, and I shift them to 0-based indices.
    return [classes[i] for i in sorted(classes)]


def load_samples(root: Path) -> list[Sample]:
    """Join images.txt, image_class_labels.txt and train_test_split.txt."""
    paths = _read_pairs(root / "images.txt")
    labels = _read_pairs(root / "image_class_labels.txt")
    split = _read_pairs(root / "train_test_split.txt")
    samples = []
    for image_id in sorted(paths):
        samples.append(
            Sample(
                image_id=image_id,
                path=paths[image_id],
                # I convert the 1-based class id to a 0-based index.
                label=int(labels[image_id]) - 1,
                # In the official split file, 1 means train and 0 means test.
                is_train=split[image_id] == "1",
            )
        )
    return samples


def stratified_val_split(
    samples: list[Sample], val_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """Hold out val_fraction of each class from the official training set."""
    rng = np.random.default_rng(seed)
    train, val = [], []
    # I group the training images by class so every class gets its share.
    by_class: dict[int, list[Sample]] = {}
    for s in samples:
        if s.is_train:
            by_class.setdefault(s.label, []).append(s)
    for label in sorted(by_class):
        group = sorted(by_class[label], key=lambda s: s.image_id)
        # I shuffle with a fixed seed so the split is the same every run.
        order = rng.permutation(len(group))
        # I round and keep at least one validation image per class.
        n_val = max(1, int(round(len(group) * val_fraction)))
        val_idx = set(order[:n_val].tolist())
        for i, s in enumerate(group):
            (val if i in val_idx else train).append(s)
    return train, val


def official_test(samples: list[Sample]) -> list[Sample]:
    return [s for s in samples if not s.is_train]


def write_split_csv(path: Path, samples: list[Sample]) -> None:
    # I write each split to CSV so the exact split is committed to the repo.
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "path", "label"])
        for s in samples:
            writer.writerow([s.image_id, s.path, s.label])


def read_split_csv(path: Path) -> list[tuple[str, int]]:
    with open(path, encoding="utf-8") as f:
        return [(row["path"], int(row["label"])) for row in csv.DictReader(f)]


def clean_class_names(raw_names: list[str]) -> list[str]:
    return [clean_label(n) for n in raw_names]
