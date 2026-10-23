"""Write the train, val and test split CSVs from the official CUB files.

Usage: python make_splits.py
"""

from collections import Counter

from birdlens_train.config import DATA_ROOT, SEED, SPLITS_DIR, VAL_FRACTION
from birdlens_train.dataset import (
    clean_class_names,
    load_class_names,
    load_samples,
    official_test,
    stratified_val_split,
    write_split_csv,
)


def main() -> None:
    samples = load_samples(DATA_ROOT)
    # I use the official train/test split and carve validation out of train only.
    train, val = stratified_val_split(samples, VAL_FRACTION, SEED)
    test = official_test(samples)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        write_split_csv(SPLITS_DIR / f"{name}.csv", split)
    # I also save the cleaned names in class index order for the app.
    names = clean_class_names(load_class_names(DATA_ROOT))
    (SPLITS_DIR / "labels.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    val_counts = Counter(s.label for s in val)
    print(f"train={len(train)} val={len(val)} test={len(test)} classes={len(names)}")
    print(f"val images per class: min={min(val_counts.values())} max={max(val_counts.values())}")


if __name__ == "__main__":
    main()
