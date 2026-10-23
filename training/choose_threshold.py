"""Pick the app's "not confident" threshold from validation data.

The rule: the lowest threshold where predictions at or above it are at least
TARGET_ACCURACY correct on the validation set. I pick it on validation, not
test, so the test numbers stay an honest check.

Usage: python choose_threshold.py --variant float16
Writes results/threshold.json and docs/threshold_curve.png.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from birdlens_train.config import DOCS_DIR, EXPORT_DIR, RESULTS_DIR  # noqa: E402
from birdlens_train.metrics import threshold_table  # noqa: E402

TARGET_ACCURACY = 0.90


def table_for(split: str, variant: str, thresholds) -> list[dict]:
    probs = np.load(EXPORT_DIR / f"{split}_probs_{variant}.npy")
    labels = np.load(EXPORT_DIR / f"{split}_labels.npy")
    return threshold_table(probs.max(axis=1), probs.argmax(axis=1) == labels, thresholds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()

    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    val = table_for("val", args.variant, thresholds)
    # I take the first (lowest) threshold that reaches the target accuracy,
    # so as many photos as possible still get a confident answer.
    chosen = next((r for r in val if r["accuracy_above"] >= TARGET_ACCURACY), None)
    if chosen is None:
        raise SystemExit(f"No threshold reaches {TARGET_ACCURACY} accuracy; rethink the rule.")
    # Then I check what that same threshold does on the test set.
    test = {r["threshold"]: r for r in table_for("test", args.variant, thresholds)}
    result = {
        "variant": args.variant,
        "rule": f"lowest threshold with validation accuracy above it >= {TARGET_ACCURACY}",
        "threshold": chosen["threshold"],
        "val": chosen,
        "test_at_same_threshold": test[chosen["threshold"]],
    }
    print(json.dumps(result, indent=2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "threshold.json").write_text(json.dumps(result, indent=2))

    # A plot of coverage and accuracy against the threshold for the docs.
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=150)
    ax.plot(thresholds, [r["accuracy_above"] for r in val], label="Accuracy at or above threshold")
    ax.plot(thresholds, [r["coverage"] for r in val], label="Share of photos at or above threshold")
    ax.axvline(chosen["threshold"], color="grey", linestyle="--", linewidth=1)
    ax.text(chosen["threshold"] + 0.01, 0.05, f"chosen {chosen['threshold']:.2f}", color="grey")
    ax.set_xlabel("Top-1 confidence threshold")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Validation set, {args.variant} model")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "threshold_curve.png")


if __name__ == "__main__":
    main()
