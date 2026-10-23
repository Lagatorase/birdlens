"""Two-stage transfer learning on CUB-200-2011, resumable from the last epoch.

Usage (inside WSL, from training/):
    nice -n 10 .venv/bin/python train.py
Run it again after an interruption and it continues where it stopped.
"""

from __future__ import annotations

import argparse
import json
import time

# I configure the GPU before anything else touches TensorFlow.
from birdlens_train.gpu import configure_gpu

GPUS = configure_gpu()

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402

from birdlens_train.config import CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR, SEED  # noqa: E402
from birdlens_train.model import build_model, unfreeze_top  # noqa: E402
from birdlens_train.pipeline import make_dataset  # noqa: E402


def stage_callbacks(stage: str, patience: int) -> list[keras.callbacks.Callback]:
    stage_dir = CHECKPOINT_DIR / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    return [
        # BackupAndRestore saves the model, optimizer and epoch at the end of
        # every epoch, and picks them up again if I rerun the script.
        keras.callbacks.BackupAndRestore(str(stage_dir / "backup")),
        # I also keep a full checkpoint for every epoch as the brief asks.
        keras.callbacks.ModelCheckpoint(str(stage_dir / "epoch_{epoch:02d}.keras")),
        # This copy only gets overwritten when validation accuracy improves.
        keras.callbacks.ModelCheckpoint(
            str(stage_dir / "best.keras"), monitor="val_accuracy", mode="max", save_best_only=True
        ),
        # I stop when validation accuracy stalls and roll back to the best epoch.
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=patience, restore_best_weights=True
        ),
        # append=True keeps the history across restarts.
        keras.callbacks.CSVLogger(str(LOG_DIR / f"{stage}_history.csv"), append=True),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-epochs", type=int, default=15)
    parser.add_argument("--finetune-epochs", type=int, default=40)
    parser.add_argument("--unfreeze-fraction", type=float, default=0.4)
    args = parser.parse_args()

    print(f"GPUs visible to TensorFlow: {GPUS}")
    if not GPUS:
        raise SystemExit("No GPU found. I stop here instead of training slowly on the CPU.")

    keras.utils.set_random_seed(SEED)
    # Mixed precision runs most maths in float16, which is faster on this GPU
    # and roughly halves activation memory.
    keras.mixed_precision.set_global_policy("mixed_float16")

    train_ds = make_dataset("train", training=True)
    val_ds = make_dataset("val", training=False)
    # Label smoothing stops the model getting overconfident, which also
    # helps the app's "not confident" threshold mean something.
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    metrics = [
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.TopKCategoricalAccuracy(k=5, name="top5"),
    ]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timings = {}

    # Stage 1: I freeze the backbone first so the new head can learn without
    # wrecking the pretrained weights with large early gradients.
    head_done = CHECKPOINT_DIR / "head" / "DONE"
    if not head_done.exists():
        model = build_model()
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=metrics)
        start = time.time()
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.head_epochs,
            callbacks=stage_callbacks("head", patience=4),
            verbose=2,
        )
        timings["head_seconds"] = round(time.time() - start)
        head_done.write_text("done\n")
    else:
        print("Stage 1 already finished, skipping.")

    # Stage 2: I start from the best head checkpoint, unfreeze the top of the
    # backbone, and fine-tune with a much smaller learning rate.
    finetune_done = CHECKPOINT_DIR / "finetune" / "DONE"
    if finetune_done.exists():
        print("Stage 2 already finished, skipping to the summary.")
    else:
        timings.update(finetune(args, train_ds, val_ds, loss, metrics))
        finetune_done.write_text("done\n")

    # I keep stage timings in their own file so a resumed run doesn't lose them.
    timings_file = RESULTS_DIR / "train_timings.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if timings_file.exists():
        timings = {**json.loads(timings_file.read_text()), **timings}
    timings_file.write_text(json.dumps(timings, indent=2))

    # I report the best validation score from the saved best checkpoint.
    # Clearing the session first matters: re-scoring inside the same process
    # that just trained gave 0.7625 instead of the 0.7717 Keras logged, while
    # a clean load of the same file reproduced 0.7717 exactly.
    keras.backend.clear_session()
    best = build_model(weights=None)
    best.load_weights(str(CHECKPOINT_DIR / "finetune" / "best.keras"))
    best.compile(loss=loss, metrics=metrics)
    val_loss, val_acc, val_top5 = best.evaluate(val_ds, verbose=0)
    summary = {
        "best_checkpoint_val_accuracy": val_acc,
        "best_checkpoint_val_top5": val_top5,
        "best_checkpoint_val_loss": val_loss,
        **{f"{stage}_{k}": v for stage in ("head", "finetune") for k, v in history_facts(stage).items()},
        **timings,
    }
    print(json.dumps(summary, indent=2))
    (RESULTS_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2))


def history_facts(stage: str) -> dict:
    """Epochs run and the best epoch, read back from the CSV log."""
    import csv

    with open(LOG_DIR / f"{stage}_history.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    best = max(rows, key=lambda r: float(r["val_accuracy"]))
    return {
        "epochs_run": len(rows),
        # CSVLogger counts epochs from 0, so I add 1 to match the console log.
        "best_epoch": int(best["epoch"]) + 1,
        "best_val_accuracy_logged": float(best["val_accuracy"]),
    }


def finetune(args, train_ds, val_ds, loss, metrics) -> dict:
    model = build_model(weights=None)
    model.load_weights(str(CHECKPOINT_DIR / "head" / "best.keras"))
    n_trainable = unfreeze_top(model, args.unfreeze_fraction)
    print(f"Fine-tuning {n_trainable} backbone layers")
    steps_per_epoch = int(tf.data.experimental.cardinality(train_ds).numpy())
    # Cosine decay lowers the learning rate smoothly so late epochs settle.
    schedule = keras.optimizers.schedules.CosineDecay(
        1e-4, decay_steps=steps_per_epoch * args.finetune_epochs, alpha=0.05
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(schedule, weight_decay=1e-5), loss=loss, metrics=metrics
    )
    start = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.finetune_epochs,
        callbacks=stage_callbacks("finetune", patience=6),
        verbose=2,
    )
    return {"finetune_seconds": round(time.time() - start)}


if __name__ == "__main__":
    main()
