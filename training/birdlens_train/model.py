"""The classifier: an ImageNet EfficientNetV2-B0 backbone with a new 200-way head."""

from __future__ import annotations

import keras

from .config import IMG_SIZE, NUM_CLASSES


def build_model(weights: str | None = "imagenet", dropout: float = 0.3) -> keras.Model:
    # The input takes raw 0 to 255 RGB pixels. EfficientNetV2 has its own
    # rescaling layer at the start, so the app never has to normalise.
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    backbone = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=weights,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
        include_preprocessing=True,
    )
    backbone.trainable = False
    # training=False keeps the batch norm layers in inference mode even after
    # I unfreeze the backbone, which is the usual advice for fine-tuning.
    x = backbone(inputs, training=False)
    x = keras.layers.Dropout(dropout, name="head_dropout")(x)
    # I keep the last layer in float32 so softmax stays stable under mixed precision.
    outputs = keras.layers.Dense(
        NUM_CLASSES, activation="softmax", dtype="float32", name="species"
    )(x)
    return keras.Model(inputs, outputs, name="birdlens_effnetv2b0")


def backbone_of(model: keras.Model) -> keras.Model:
    # The backbone is the only nested model inside mine.
    return next(layer for layer in model.layers if isinstance(layer, keras.Model))


def unfreeze_top(model: keras.Model, fraction: float) -> int:
    """Unfreeze the top fraction of backbone layers, leaving batch norm frozen."""
    backbone = backbone_of(model)
    backbone.trainable = True
    layers = backbone.layers
    cut = int(len(layers) * (1.0 - fraction))
    trainable = 0
    for i, layer in enumerate(layers):
        # I keep the lower layers frozen, since they hold generic edge and
        # texture features that transfer well to birds as they are.
        # Batch norm stays frozen too, because its running statistics are
        # fragile with a small batch of 16.
        is_bn = isinstance(layer, keras.layers.BatchNormalization)
        layer.trainable = i >= cut and not is_bn
        trainable += int(layer.trainable)
    return trainable
