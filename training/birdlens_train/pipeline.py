"""tf.data pipelines built from the committed split CSVs."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from .config import BATCH_SIZE, DATA_ROOT, DATA_WORKERS, NUM_CLASSES, SEED, SPLITS_DIR
from .dataset import read_split_csv
from .preprocess import eval_preprocess, train_augment


def split_paths(name: str, root: Path = DATA_ROOT) -> tuple[list[str], list[int]]:
    rows = read_split_csv(SPLITS_DIR / f"{name}.csv")
    paths = [str(root / "images" / p) for p, _ in rows]
    labels = [label for _, label in rows]
    return paths, labels


def make_dataset(name: str, training: bool, batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    paths, labels = split_paths(name)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        # I reshuffle every epoch with a fixed seed so runs are repeatable.
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    fn = train_augment if training else eval_preprocess

    def load(path, label):
        image = fn(tf.io.read_file(path))
        # I use one-hot labels so I can apply label smoothing in the loss.
        return image, tf.one_hot(label, NUM_CLASSES)

    ds = ds.map(load, num_parallel_calls=DATA_WORKERS)
    ds = ds.batch(batch_size).prefetch(2)
    # I cap tf.data's private thread pool so it doesn't take every core.
    options = tf.data.Options()
    options.threading.private_threadpool_size = DATA_WORKERS
    return ds.with_options(options)
