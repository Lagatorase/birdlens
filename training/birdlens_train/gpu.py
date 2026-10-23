"""Keeping TensorFlow polite on a GPU I'm also gaming on."""

import os

from .config import GPU_MEMORY_LIMIT_MB


def configure_gpu(memory_limit_mb: int = GPU_MEMORY_LIMIT_MB) -> list:
    """Enable memory growth and cap TensorFlow's VRAM use. Call before any model code."""
    # I set growth through the env var because TensorFlow refuses
    # set_memory_growth once a virtual device with a memory limit exists.
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        # I give TensorFlow one logical GPU capped at about half the card.
        tf.config.set_logical_device_configuration(
            gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
        )
    return gpus
