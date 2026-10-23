import io

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from PIL import Image  # noqa: E402

from birdlens_train.preprocess import eval_preprocess, train_augment  # noqa: E402


def _png_bytes(array: np.ndarray) -> bytes:
    # I use PNG so the pixels survive encoding exactly.
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


def _reference_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    # My own half-pixel-centre bilinear resize, written the same way as the
    # Dart version, so this test pins down what the app has to reproduce.
    in_h, in_w, _ = img.shape
    img = img.astype(np.float64)
    out = np.zeros((out_h, out_w, 3))
    for y in range(out_h):
        sy = max((y + 0.5) * in_h / out_h - 0.5, 0.0)
        y0 = min(int(np.floor(sy)), in_h - 1)
        y1 = min(y0 + 1, in_h - 1)
        dy = sy - y0
        for x in range(out_w):
            sx = max((x + 0.5) * in_w / out_w - 0.5, 0.0)
            x0 = min(int(np.floor(sx)), in_w - 1)
            x1 = min(x0 + 1, in_w - 1)
            dx = sx - x0
            top = img[y0, x0] * (1 - dx) + img[y0, x1] * dx
            bottom = img[y1, x0] * (1 - dx) + img[y1, x1] * dx
            out[y, x] = top * (1 - dy) + bottom * dy
    return out


def test_eval_output_shape_and_range():
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(300, 400, 3), dtype=np.uint8)
    out = eval_preprocess(tf.constant(_png_bytes(array))).numpy()
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.float32
    # Pixels stay in 0 to 255 because the model does its own rescaling.
    assert out.min() >= 0.0 and out.max() <= 255.0


def test_eval_matches_reference_bilinear():
    rng = np.random.default_rng(1)
    array = rng.integers(0, 256, size=(375, 500, 3), dtype=np.uint8)
    out = eval_preprocess(tf.constant(_png_bytes(array))).numpy()
    ref = _reference_bilinear(array, 224, 224)
    # This is the exact algorithm the Dart code implements.
    np.testing.assert_allclose(out, ref, atol=1e-3)


def test_greyscale_becomes_rgb():
    array = np.full((50, 60), 128, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(array, mode="L").save(buf, format="PNG")
    out = eval_preprocess(tf.constant(buf.getvalue())).numpy()
    assert out.shape == (224, 224, 3)
    np.testing.assert_allclose(out, 128.0, atol=1e-4)


def test_train_augment_shape_and_range():
    rng = np.random.default_rng(2)
    array = rng.integers(0, 256, size=(300, 400, 3), dtype=np.uint8)
    tf.random.set_seed(0)
    out = train_augment(tf.constant(_png_bytes(array))).numpy()
    assert out.shape == (224, 224, 3)
    assert out.min() >= 0.0 and out.max() <= 255.0
