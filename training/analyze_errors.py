"""Error analysis for one TFLite variant on the CUB test set.

Usage: python analyze_errors.py --variant float16
Needs evaluate.py to have run first. Writes results/error_analysis.json and
docs/misclassifications.png.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from birdlens_train.config import DATA_ROOT, DOCS_DIR, EXPORT_DIR, NUM_CLASSES, RESULTS_DIR, SPLITS_DIR  # noqa: E402
from birdlens_train.dataset import read_split_csv  # noqa: E402
from birdlens_train.metrics import confusion_matrix, most_confused_pairs, per_class_accuracy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()

    names = (SPLITS_DIR / "labels.txt").read_text(encoding="utf-8").splitlines()
    probs = np.load(EXPORT_DIR / f"test_probs_{args.variant}.npy")
    labels = np.load(EXPORT_DIR / "test_labels.npy")
    paths = [p for p, _ in read_split_csv(SPLITS_DIR / "test.csv")]
    preds = probs.argmax(axis=1)
    cm = confusion_matrix(labels, preds, NUM_CLASSES)

    # The ten classes with the lowest test accuracy.
    acc = per_class_accuracy(cm)
    worst = np.argsort(acc, kind="stable")[:10]
    worst_rows = [
        {
            "class": names[c],
            "accuracy": float(acc[c]),
            "test_images": int(cm[c].sum()),
            # What it gets confused with most, which usually explains the score.
            "most_common_wrong_guess": names[int(np.argmax(np.where(np.arange(NUM_CLASSES) == c, -1, cm[c])))],
        }
        for c in worst
    ]

    # The ten pairs of species mixed up most often, in either direction.
    pairs = most_confused_pairs(cm, 10)
    pair_rows = [
        {
            "species_a": names[a],
            "species_b": names[b],
            "a_called_b": ab,
            "b_called_a": ba,
            "total": ab + ba,
        }
        for a, b, ab, ba in pairs
    ]
    result = {"variant": args.variant, "worst_classes": worst_rows, "most_confused_pairs": pair_rows}
    print(json.dumps(result, indent=2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "error_analysis.json").write_text(json.dumps(result, indent=2))

    # A grid of real mistakes from the most confused pairs. For each pair I
    # show the most confident wrong prediction, since those are the most telling.
    examples = []
    for a, b, _, _ in pairs:
        for true, wrong in ((a, b), (b, a)):
            idx = np.where((labels == true) & (preds == wrong))[0]
            if len(idx):
                best = idx[np.argmax(probs[idx, wrong])]
                examples.append((best, true, wrong))
            if len(examples) >= 12:
                break
        if len(examples) >= 12:
            break
    cols = 4
    rows = int(np.ceil(len(examples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4), dpi=110)
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (i, true, wrong) in zip(np.atleast_1d(axes).ravel(), examples):
        image = Image.open(DATA_ROOT / "images" / paths[i]).convert("RGB")
        # I shrink each photo for the grid. These are a few CUB test images
        # shown for analysis only, not the dataset itself.
        image.thumbnail((300, 300))
        ax.imshow(image)
        ax.set_title(
            f"True: {names[true]}\nGuess: {names[wrong]} ({probs[i, wrong]:.0%})",
            fontsize=7.5,
        )
    fig.suptitle(f"Confident mistakes from the most confused pairs ({args.variant} model)", fontsize=10)
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "misclassifications.png")


if __name__ == "__main__":
    main()
