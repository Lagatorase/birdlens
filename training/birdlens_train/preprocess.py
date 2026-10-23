"""Image preprocessing for training and evaluation.

The eval path is the contract with the Flutter app: decode as RGB, resize the
whole image to 224x224 with plain bilinear sampling (half-pixel centres, no
antialiasing), and keep pixel values in the 0 to 255 range. The model rescales
internally, so the app does no normalisation of its own.
"""

from __future__ import annotations

import tensorflow as tf

from .config import IMG_SIZE


def decode_rgb(jpeg_bytes: tf.Tensor) -> tf.Tensor:
    # A few CUB images are greyscale, so I force three channels.
    return tf.io.decode_image(jpeg_bytes, channels=3, expand_animations=False)


def resize_for_model(image: tf.Tensor) -> tf.Tensor:
    """Squash the full image to IMG_SIZE x IMG_SIZE with bilinear sampling."""
    # I resize the whole frame instead of centre cropping, because birds are
    # often off centre and a crop can cut off the head or tail.
    # antialias=False keeps this a simple bilinear sample that Dart can copy.
    image = tf.image.resize(
        tf.cast(image, tf.float32), (IMG_SIZE, IMG_SIZE), method="bilinear", antialias=False
    )
    return image


def eval_preprocess(jpeg_bytes: tf.Tensor) -> tf.Tensor:
    """The exact preprocessing the app reproduces in Dart."""
    return resize_for_model(decode_rgb(jpeg_bytes))


def train_augment(jpeg_bytes: tf.Tensor) -> tf.Tensor:
    """Random crop, flip and mild colour jitter for training images."""
    image = decode_rgb(jpeg_bytes)
    # I take a random crop that keeps most of the frame, so the bird usually
    # stays in view while the model sees different scales and positions.
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=tf.zeros([1, 0, 4]),
        min_object_covered=0.0,
        aspect_ratio_range=(3 / 4, 4 / 3),
        area_range=(0.5, 1.0),
        max_attempts=20,
        use_image_if_no_bounding_boxes=True,
    )
    image = tf.slice(image, begin, size)
    image = resize_for_model(image)
    # Birds look the same mirrored left to right, but never upside down,
    # so I only flip horizontally.
    image = tf.image.random_flip_left_right(image)
    # I keep the colour jitter mild because plumage colour is a key clue.
    image = image / 255.0
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.image.random_saturation(image, 0.9, 1.1)
    image = tf.image.random_hue(image, 0.02)
    # I clip back into range and return to the 0 to 255 scale the model expects.
    return tf.clip_by_value(image, 0.0, 1.0) * 255.0
