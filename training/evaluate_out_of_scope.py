"""How the shipped float16 model behaves on photos it was never meant to see.

I run the app's .tflite file through the LiteRT interpreter on the CPU on three
sets: the CUB test set (for reference), birds of species outside CUB, and photos
with no birds at all. For each set I report how often the top-1 probability is
under the app's threshold, which is when the app says "Not confident".
The threshold stays the one picked on validation. Nothing here changes it.

Usage (from training/, after scripts/build_out_of_scope.py):
    python evaluate_out_of_scope.py
Writes results/out_of_scope.json and docs/out_of_scope_confidence.png.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from birdlens_train.config import APP_ASSETS_DIR, DOCS_DIR, RESULTS_DIR, WORK_DIR  # noqa: E402
from birdlens_train.metrics import topk_accuracy  # noqa: E402
from birdlens_train.pipeline import split_paths  # noqa: E402
from birdlens_train.preprocess import eval_preprocess  # noqa: E402
from birdlens_train.tflite_utils import TfliteModel  # noqa: E402

# Like evaluate.py, I hide the GPU because TensorFlow only decodes and resizes here.
tf.config.set_visible_devices([], "GPU")

MODEL = APP_ASSETS_DIR / "models" / "birdlens.tflite"
MANIFEST_DIR = Path(__file__).resolve().parent / "out_of_scope"
IMAGE_DIR = WORK_DIR / "out_of_scope"
BATCH = 32
# The same palette slots in every figure, one per set.
COLORS = {"cub_test": "#2a78d6", "non_cub_birds": "#eb6834", "non_birds": "#1baf7a"}
TITLES = {
    "cub_test": "CUB test set (the 200 species)",
    "non_cub_birds": "Birds of species not in CUB",
    "non_birds": "Photos with no bird",
}


def predict(model: TfliteModel, paths: list[str]) -> np.ndarray:
    # I use the same tf.data preprocessing as evaluate.py, so the JPEG decoder matches too.
    ds = (
        tf.data.Dataset.from_tensor_slices(paths)
        .map(lambda p: eval_preprocess(tf.io.read_file(p)), num_parallel_calls=4)
        .batch(BATCH)
    )
    return np.concatenate([model.predict(batch.numpy()) for batch in ds])


def summarise(confidence: np.ndarray, threshold: float) -> dict:
    return {
        "images": int(len(confidence)),
        "not_confident_rate": float(np.mean(confidence < threshold)),
        "confidence_median": float(np.median(confidence)),
        "confidence_mean": float(np.mean(confidence)),
        "confidence_p10": float(np.percentile(confidence, 10)),
        "confidence_p90": float(np.percentile(confidence, 90)),
        "share_at_or_above_0.9": float(np.mean(confidence >= 0.9)),
    }


def plot(confidence: dict[str, np.ndarray], summary: dict, threshold: float) -> None:
    bins = np.linspace(0, 1, 21)
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.2), dpi=150, sharex=True)
    for ax, name in zip(axes, TITLES):
        conf = confidence[name]
        # I plot percentages so sets of different sizes are comparable.
        weights = np.full(len(conf), 100 / len(conf))
        ax.hist(conf, bins=bins, weights=weights, color=COLORS[name], rwidth=0.88)
        ax.axvline(threshold, color="#3d3d3a", linewidth=1.2, linestyle="--")
        rate = summary[name]["not_confident_rate"]
        ax.set_title(
            f"{TITLES[name]}  (n = {len(conf):,})", loc="left", fontsize=9.5, color="#1f1f1d"
        )
        ax.text(
            threshold - 0.01, 0.93, f"{rate:.0%} under {threshold}\n(\"Not confident\")",
            transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=8, color="#55554f",
        )
        ax.set_ylabel("% of photos", fontsize=8.5, color="#55554f")
        ax.tick_params(labelsize=8, colors="#55554f")
        ax.grid(axis="y", color="#e4e3dd", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
    axes[-1].set_xlabel("Top-1 probability from the float16 model", fontsize=8.5, color="#55554f")
    axes[-1].set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "out_of_scope_confidence.png", facecolor="white")
    plt.close(fig)


def main() -> None:
    threshold = json.loads((RESULTS_DIR / "threshold.json").read_text())["threshold"]
    labels = (APP_ASSETS_DIR / "labels.txt").read_text(encoding="utf-8").splitlines()
    model = TfliteModel(MODEL, threads=4, batch_size=BATCH)

    # The CUB test set, for reference only.
    cub_paths, cub_labels = split_paths("test")
    cub_probs = predict(model, cub_paths)
    cub_labels = np.array(cub_labels)
    confidence = {"cub_test": cub_probs.max(axis=1)}
    summary = {"cub_test": summarise(confidence["cub_test"], threshold)}
    # This should repeat evaluate.py's float16 test top-1 in tflite_metrics.json.
    summary["cub_test"]["top1"] = topk_accuracy(cub_probs, cub_labels, 1)

    for name in ("non_cub_birds", "non_birds"):
        with open(MANIFEST_DIR / f"{name}.csv", encoding="utf-8") as f:
            manifest = list(csv.DictReader(f))
        probs = predict(model, [str(IMAGE_DIR / name / r["file"]) for r in manifest])
        conf = probs.max(axis=1)
        confidence[name] = conf
        summary[name] = summarise(conf, threshold)
        # I break the rate down by group and by class to see where it fails.
        for key in ("group", "name"):
            parts = {}
            for value in dict.fromkeys(r[key] for r in manifest):
                mask = np.array([r[key] == value for r in manifest])
                parts[value] = {
                    "images": int(mask.sum()),
                    "not_confident_rate": float(np.mean(conf[mask] < threshold)),
                    "confidence_median": float(np.median(conf[mask])),
                }
            summary[name][f"by_{key}"] = parts
        # I also note which CUB species the confident answers go to.
        confident = conf >= threshold
        top = Counter(labels[i] for i in probs.argmax(axis=1)[confident]).most_common(5)
        summary[name]["most_common_confident_answers"] = top

    result = {"model": "app/assets/models/birdlens.tflite (float16)", "threshold": threshold, **summary}
    print(json.dumps(result, indent=2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "out_of_scope.json").write_text(json.dumps(result, indent=2))
    plot(confidence, summary, threshold)


if __name__ == "__main__":
    main()
