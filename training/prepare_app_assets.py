"""Copy the chosen model into the Flutter app and build the parity fixtures.

Usage: python prepare_app_assets.py --variant float16
Needs evaluate.py and choose_threshold.py to have run first.

Writes:
  app/assets/models/birdlens.tflite, app/assets/labels.txt, app/assets/model_config.json
  app/assets/samples/sample_*.jpg       (benchmark photos)
  app/test/fixtures/                    (fixture images and expected outputs)
  app/integration_test/fixture_data.dart (the same fixtures embedded for the device test)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil

import numpy as np
import tensorflow as tf
from PIL import Image

from birdlens_train.config import (
    APP_ASSETS_DIR,
    APP_FIXTURES_DIR,
    DATA_ROOT,
    EXPORT_DIR,
    IMG_SIZE,
    NUM_CLASSES,
    REPO_DIR,
    RESULTS_DIR,
    SEED,
    SPLITS_DIR,
)
from birdlens_train.dataset import read_split_csv
from birdlens_train.preprocess import eval_preprocess
from birdlens_train.tflite_utils import VARIANTS, TfliteModel

tf.config.set_visible_devices([], "GPU")

N_FIXTURES = 4
N_SAMPLES = 3
FIXTURE_MAX_SIDE = 320
# Fixed pixel positions I compare between Python and Dart.
N_PROBE_POINTS = 64


def pick_test_images(n: int, skip: set[int]) -> list[tuple[int, str, int]]:
    rows = read_split_csv(SPLITS_DIR / "test.csv")
    rng = np.random.default_rng(SEED)
    picked, classes = [], set()
    # I pick images from different classes so the fixtures cover a spread.
    for i in rng.permutation(len(rows)):
        path, label = rows[i]
        if i in skip or label in classes:
            continue
        picked.append((int(i), path, label))
        classes.add(label)
        if len(picked) == n:
            break
    return picked


def small_copies(path: str) -> tuple[bytes, bytes]:
    """A downsized PNG and JPEG of one CUB test image."""
    image = Image.open(DATA_ROOT / "images" / path).convert("RGB")
    image.thumbnail((FIXTURE_MAX_SIDE, FIXTURE_MAX_SIDE), Image.Resampling.LANCZOS)
    png, jpg = io.BytesIO(), io.BytesIO()
    image.save(png, format="PNG")
    image.save(jpg, format="JPEG", quality=92)
    return png.getvalue(), jpg.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    args = parser.parse_args()

    metrics = json.loads((RESULTS_DIR / "tflite_metrics.json").read_text())[args.variant]
    threshold = json.loads((RESULTS_DIR / "threshold.json").read_text())
    assert threshold["variant"] == args.variant, "threshold was chosen for a different variant"

    # 1. The model, labels and config the app loads.
    (APP_ASSETS_DIR / "models").mkdir(parents=True, exist_ok=True)
    model_path = EXPORT_DIR / f"birdlens_{args.variant}.tflite"
    shutil.copy(model_path, APP_ASSETS_DIR / "models" / "birdlens.tflite")
    shutil.copy(SPLITS_DIR / "labels.txt", APP_ASSETS_DIR / "labels.txt")
    config = {
        "variant": args.variant,
        "model_asset": "assets/models/birdlens.tflite",
        "labels_asset": "assets/labels.txt",
        "input_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "threshold": threshold["threshold"],
        "test_top1": round(metrics["test_top1"], 4),
        "test_top5": round(metrics["test_top5"], 4),
    }
    (APP_ASSETS_DIR / "model_config.json").write_text(json.dumps(config, indent=2) + "\n")

    # 2. Fixture images and what the Python TFLite runtime says about them.
    fixtures = pick_test_images(N_FIXTURES, skip=set())
    samples = pick_test_images(N_SAMPLES, skip={i for i, _, _ in fixtures})
    images_dir = APP_FIXTURES_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    models = {v: TfliteModel(EXPORT_DIR / f"birdlens_{v}.tflite") for v in VARIANTS}
    rng = np.random.default_rng(SEED)
    probes = rng.integers(0, IMG_SIZE, size=(N_PROBE_POINTS, 2)).tolist()
    entries = []
    embedded = {}
    for n, (_, path, label) in enumerate(fixtures, start=1):
        png, jpg = small_copies(path)
        name = f"fixture_{n}"
        (images_dir / f"{name}.png").write_bytes(png)
        (images_dir / f"{name}.jpg").write_bytes(jpg)
        embedded[f"{name}.png"] = png
        embedded[f"{name}.jpg"] = jpg
        entry = {"name": name, "true_label": label, "inputs": {}}
        for kind, data in (("png", png), ("jpg", jpg)):
            # This is exactly the preprocessing the app copies in Dart.
            pixels = eval_preprocess(tf.constant(data)).numpy()
            outputs = {}
            for variant, model in models.items():
                probs = model.predict(pixels[None])[0]
                top5 = np.argsort(probs)[::-1][:5]
                outputs[variant] = {
                    "top5": top5.tolist(),
                    "top5_probs": [round(float(probs[i]), 6) for i in top5],
                    # The full vector for the shipped variant, for a strict comparison.
                    **({"probs": [round(float(p), 6) for p in probs]} if variant == args.variant else {}),
                }
            entry["inputs"][kind] = {
                "channel_means": [round(float(m), 4) for m in pixels.mean(axis=(0, 1))],
                "probe_values": [[round(float(v), 4) for v in pixels[y, x]] for y, x in probes],
                "outputs": outputs,
            }
        entries.append(entry)
    fixture_doc = {
        "note": "Downsized CUB-200-2011 test images (non-commercial research use) and the "
        "outputs of the Python preprocessing and LiteRT interpreter for each model variant.",
        "shipped_variant": args.variant,
        "input_size": IMG_SIZE,
        "probe_points_yx": probes,
        "fixtures": entries,
    }
    (APP_FIXTURES_DIR / "fixtures.json").write_text(json.dumps(fixture_doc, indent=1))

    # 3. The same data embedded in Dart, because the device test can't read
    # files from my PC.
    lines = [
        "// Generated by training/prepare_app_assets.py. Do not edit by hand.",
        "// ignore_for_file: lines_longer_than_80_chars",
        "",
        "const fixturesJson = r'''",
        json.dumps(fixture_doc),
        "''';",
        "",
        "const fixtureImagesBase64 = <String, String>{",
        *[f"  '{k}': '{base64.b64encode(v).decode()}'," for k, v in embedded.items()],
        "};",
        "",
    ]
    (REPO_DIR / "app" / "integration_test").mkdir(exist_ok=True)
    (REPO_DIR / "app" / "integration_test" / "fixture_data.dart").write_text("\n".join(lines))

    # 4. Benchmark photos for the app, also downsized.
    samples_dir = APP_ASSETS_DIR / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for n, (_, path, _) in enumerate(samples, start=1):
        _, jpg = small_copies(path)
        (samples_dir / f"sample_{n}.jpg").write_bytes(jpg)

    print(json.dumps(config, indent=2))
    print(f"fixtures: {[e['name'] for e in entries]}, samples: {len(samples)}")


if __name__ == "__main__":
    main()
