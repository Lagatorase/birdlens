"""Evaluate every .tflite variant on the official CUB test set with the LiteRT interpreter.

Also scores the validation set, which choose_threshold.py uses.
Usage (inside WSL, from training/): python evaluate.py
Writes results/tflite_metrics.json and saves the raw probabilities to the work folder.
"""

from __future__ import annotations

import json
import time

import numpy as np
import tensorflow as tf

from birdlens_train.config import EXPORT_DIR, NUM_CLASSES, RESULTS_DIR
from birdlens_train.metrics import confusion_matrix, macro_f1, topk_accuracy
from birdlens_train.pipeline import split_paths
from birdlens_train.preprocess import eval_preprocess
from birdlens_train.tflite_utils import VARIANTS, TfliteModel

# I hide the GPU from TensorFlow here: it only decodes and resizes images.
tf.config.set_visible_devices([], "GPU")

BATCH = 32


def score_split(split: str, models: dict[str, TfliteModel]) -> tuple[dict, np.ndarray, dict]:
    """Stream one split through every model, so each image is preprocessed once."""
    paths, labels = split_paths(split)
    ds = (
        tf.data.Dataset.from_tensor_slices(paths)
        .map(lambda p: eval_preprocess(tf.io.read_file(p)), num_parallel_calls=4)
        .batch(BATCH)
    )
    probs = {v: [] for v in models}
    seconds = {v: 0.0 for v in models}
    for batch in ds:
        pixels = batch.numpy()
        for variant, model in models.items():
            start = time.time()
            probs[variant].append(model.predict(pixels))
            seconds[variant] += time.time() - start
    return {v: np.concatenate(p) for v, p in probs.items()}, np.array(labels), seconds


def main() -> None:
    models = {
        v: TfliteModel(EXPORT_DIR / f"birdlens_{v}.tflite", threads=4, batch_size=BATCH)
        for v in VARIANTS
    }
    val_probs, val_labels, _ = score_split("val", models)
    test_probs, test_labels, seconds = score_split("test", models)
    np.save(EXPORT_DIR / "val_labels.npy", val_labels)
    np.save(EXPORT_DIR / "test_labels.npy", test_labels)
    results = {}
    for variant in VARIANTS:
        np.save(EXPORT_DIR / f"val_probs_{variant}.npy", val_probs[variant])
        np.save(EXPORT_DIR / f"test_probs_{variant}.npy", test_probs[variant])
        probs = test_probs[variant]
        cm = confusion_matrix(test_labels, probs.argmax(axis=1), NUM_CLASSES)
        results[variant] = {
            "file_bytes": (EXPORT_DIR / f"birdlens_{variant}.tflite").stat().st_size,
            "input_dtype": np.dtype(models[variant].input_dtype).name,
            "test_top1": topk_accuracy(probs, test_labels, 1),
            "test_top5": topk_accuracy(probs, test_labels, 5),
            "test_macro_f1": macro_f1(cm),
            "val_top1": topk_accuracy(val_probs[variant], val_labels, 1),
            "test_images": int(len(test_labels)),
            "python_cpu_seconds_for_test_set": round(seconds[variant], 1),
        }
    # I also record how often the quantised variants agree with float32.
    base = test_probs["float32"].argmax(axis=1)
    for variant in VARIANTS[1:]:
        agree = np.mean(base == test_probs[variant].argmax(axis=1))
        results[variant]["top1_agreement_with_float32"] = float(agree)
    for variant, r in results.items():
        print(variant, json.dumps(r))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tflite_metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
