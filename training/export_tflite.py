"""Convert the best fine-tuned checkpoint into float32, float16 and int8 .tflite files.

Usage (inside WSL, from training/): python export_tflite.py
"""

from __future__ import annotations

import numpy as np

from birdlens_train.gpu import configure_gpu

# Export runs on the CPU side mostly, but I still keep the GPU capped.
configure_gpu()

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402

from birdlens_train.config import CHECKPOINT_DIR, EXPORT_DIR, SEED  # noqa: E402
from birdlens_train.model import build_model  # noqa: E402
from birdlens_train.pipeline import split_paths  # noqa: E402
from birdlens_train.preprocess import eval_preprocess  # noqa: E402

REPRESENTATIVE_IMAGES = 300


def load_float32_model() -> keras.Model:
    # Training used mixed precision. I rebuild the model in plain float32 so
    # the exported graph has no float16 casts, then copy the trained weights in.
    keras.mixed_precision.set_global_policy("float32")
    model = build_model(weights=None)
    model.load_weights(str(CHECKPOINT_DIR / "finetune" / "best.keras"))
    return model


def representative_dataset():
    """Training images for int8 calibration, preprocessed exactly like inference."""
    paths, _ = split_paths("train")
    rng = np.random.default_rng(SEED)
    # A random spread of training images covers the range of activations well.
    for i in rng.choice(len(paths), REPRESENTATIVE_IMAGES, replace=False):
        image = eval_preprocess(tf.io.read_file(paths[i]))
        yield [tf.expand_dims(image, 0)]


def convert(model: keras.Model, variant: str) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if variant == "float16":
        # Weights stored as float16 halve the file size. The runtime upcasts
        # them to float32 on CPU, so accuracy should barely move.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif variant == "int8":
        # Full integer quantisation: weights and activations in int8, with a
        # uint8 input and output so the app can feed raw pixels.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    return converter.convert()


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_float32_model()
    for variant in ("float32", "float16", "int8"):
        data = convert(model, variant)
        path = EXPORT_DIR / f"birdlens_{variant}.tflite"
        path.write_bytes(data)
        print(f"{variant}: {len(data) / 2**20:.2f} MiB -> {path}")


if __name__ == "__main__":
    main()
