"""Re-score saved checkpoints on the validation set to make sure the reloaded
weights reproduce the accuracy Keras logged during training.

Usage: python scripts/check_checkpoints.py epoch_15 epoch_19 best
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from birdlens_train.gpu import configure_gpu  # noqa: E402

configure_gpu()

import keras  # noqa: E402

from birdlens_train.config import CHECKPOINT_DIR  # noqa: E402
from birdlens_train.model import build_model  # noqa: E402
from birdlens_train.pipeline import make_dataset  # noqa: E402

policy = sys.argv[1]
val_ds = make_dataset("val", training=False)
keras.mixed_precision.set_global_policy(policy)
for name in sys.argv[2:]:
    model = build_model(weights=None)
    model.load_weights(str(CHECKPOINT_DIR / "finetune" / f"{name}.keras"))
    model.compile(metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")], loss="categorical_crossentropy")
    _, acc = model.evaluate(val_ds, verbose=0)
    print(f"{policy} {name}: val_accuracy={acc:.4f}")
