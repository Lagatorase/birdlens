from collections import Counter
from pathlib import Path

import pytest

from birdlens_train.dataset import (
    load_class_names,
    load_samples,
    official_test,
    stratified_val_split,
)


@pytest.fixture
def fake_cub(tmp_path: Path) -> Path:
    # I build a tiny fake CUB folder: 3 classes, 20 images each,
    # with the first 12 of each class in the official train split.
    classes, images, labels, split = [], [], [], []
    image_id = 1
    for c in range(1, 4):
        classes.append(f"{c} {c:03d}.Bird_{c}")
        for i in range(20):
            images.append(f"{image_id} {c:03d}.Bird_{c}/img_{i}.jpg")
            labels.append(f"{image_id} {c}")
            split.append(f"{image_id} {1 if i < 12 else 0}")
            image_id += 1
    for name, lines in [
        ("classes.txt", classes),
        ("images.txt", images),
        ("image_class_labels.txt", labels),
        ("train_test_split.txt", split),
    ]:
        (tmp_path / name).write_text("\n".join(lines) + "\n")
    return tmp_path


def test_load_samples_uses_official_split(fake_cub):
    samples = load_samples(fake_cub)
    assert len(samples) == 60
    assert sum(s.is_train for s in samples) == 36
    assert len(official_test(samples)) == 24
    # Labels are shifted to 0-based indices.
    assert {s.label for s in samples} == {0, 1, 2}


def test_class_names_in_order(fake_cub):
    assert load_class_names(fake_cub) == ["001.Bird_1", "002.Bird_2", "003.Bird_3"]


def test_val_split_is_stratified_and_disjoint(fake_cub):
    samples = load_samples(fake_cub)
    train, val = stratified_val_split(samples, 0.10, seed=42)
    # 10 percent of 12 rounds to 1 image per class.
    assert Counter(s.label for s in val) == {0: 1, 1: 1, 2: 1}
    assert len(train) + len(val) == 36
    # No image ends up in both train and val, and no test image leaks in.
    assert not {s.image_id for s in train} & {s.image_id for s in val}
    assert all(s.is_train for s in train + val)


def test_val_split_is_deterministic(fake_cub):
    samples = load_samples(fake_cub)
    a = stratified_val_split(samples, 0.25, seed=42)[1]
    b = stratified_val_split(samples, 0.25, seed=42)[1]
    c = stratified_val_split(samples, 0.25, seed=7)[1]
    assert [s.image_id for s in a] == [s.image_id for s in b]
    # A different seed should give a different split.
    assert [s.image_id for s in a] != [s.image_id for s in c]


def test_real_split_files_if_present():
    # This checks the committed split CSVs once make_splits.py has run.
    from birdlens_train.config import SPLITS_DIR
    from birdlens_train.dataset import read_split_csv

    if not (SPLITS_DIR / "train.csv").exists():
        pytest.skip("split CSVs not generated yet")
    train = read_split_csv(SPLITS_DIR / "train.csv")
    val = read_split_csv(SPLITS_DIR / "val.csv")
    test = read_split_csv(SPLITS_DIR / "test.csv")
    # The official CUB split is 5994 train and 5794 test images.
    assert len(train) + len(val) == 5994
    assert len(test) == 5794
    assert not {p for p, _ in train} & {p for p, _ in val}
    assert not {p for p, _ in train + val} & {p for p, _ in test}
    assert len({label for _, label in val}) == 200
