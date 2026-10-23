"""Quick check that TensorFlow can really train on the GPU before the long run.

Usage: python scripts/gpu_check.py
"""

import sys
import time
from pathlib import Path

# I make the package importable when this runs from training/scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from birdlens_train.gpu import configure_gpu  # noqa: E402

gpus = configure_gpu()

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402

print("TensorFlow", tf.__version__, "Keras", keras.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda(), tf.sysconfig.get_build_info().get("cuda_version"))
print("Physical GPUs:", gpus)
for gpu in gpus:
    print("Details:", tf.config.experimental.get_device_details(gpu))
if not gpus:
    raise SystemExit("No GPU visible")

# I run a real conv net step under mixed precision, which is what training does.
keras.mixed_precision.set_global_policy("mixed_float16")
model = keras.applications.EfficientNetV2B0(weights=None, classes=10)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
x = tf.random.uniform((16, 224, 224, 3), 0, 255)
y = tf.random.uniform((16,), 0, 10, dtype=tf.int32)
model.train_on_batch(x, y)
start = time.time()
for _ in range(10):
    model.train_on_batch(x, y)
print(f"10 training steps of batch 16: {time.time() - start:.2f}s")
info = tf.config.experimental.get_memory_info("GPU:0")
print(f"Peak GPU memory used by TensorFlow: {info['peak'] / 2**20:.0f} MiB")
