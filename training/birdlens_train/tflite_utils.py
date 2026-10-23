"""Running exported .tflite files with the LiteRT interpreter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter

VARIANTS = ("float32", "float16", "int8")


def quantize_like_dart(x: np.ndarray, scale: float, zero_point: int, dtype) -> np.ndarray:
    """Quantise pixels the same way the app's Dart code does.

    Dart's round() goes half away from zero, while np.round goes half to even,
    so I write the rounding out by hand to keep the two in step.
    """
    scaled = x / scale
    rounded = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    info = np.iinfo(dtype)
    return np.clip(rounded + zero_point, info.min, info.max).astype(dtype)


class TfliteModel:
    """A small wrapper that handles float and integer models the same way."""

    def __init__(self, path: Path, threads: int = 4, batch_size: int = 1):
        self.path = Path(path)
        self.interpreter = Interpreter(model_path=str(path), num_threads=threads)
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.batch_size = batch_size
        if batch_size != 1:
            # The export keeps a dynamic batch dimension, so I can run batches here.
            self.interpreter.resize_tensor_input(
                self.input["index"], [batch_size, *self.input["shape"][1:]]
            )
        self.interpreter.allocate_tensors()
        # Refresh the details after resizing.
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]

    @property
    def input_dtype(self):
        return self.input["dtype"]

    def predict(self, pixels: np.ndarray) -> np.ndarray:
        """pixels: float32 [n, 224, 224, 3] in 0 to 255. Returns probabilities [n, classes]."""
        n = len(pixels)
        out = []
        for start in range(0, n, self.batch_size):
            batch = pixels[start : start + self.batch_size]
            real = len(batch)
            # I pad the last batch to the fixed batch size and drop the padding after.
            if real < self.batch_size:
                pad = np.zeros((self.batch_size - real, *batch.shape[1:]), batch.dtype)
                batch = np.concatenate([batch, pad])
            self.interpreter.set_tensor(self.input["index"], self._prepare(batch))
            self.interpreter.invoke()
            out.append(self._read()[:real])
        return np.concatenate(out)

    def _prepare(self, batch: np.ndarray) -> np.ndarray:
        dtype = self.input["dtype"]
        if dtype == np.float32:
            return batch.astype(np.float32)
        scale, zero_point = self.input["quantization"]
        return quantize_like_dart(batch, scale, zero_point, dtype)

    def _read(self) -> np.ndarray:
        raw = self.interpreter.get_tensor(self.output["index"])
        if self.output["dtype"] == np.float32:
            return raw.astype(np.float32)
        # Integer outputs get dequantised back to probabilities.
        scale, zero_point = self.output["quantization"]
        return (scale * (raw.astype(np.float32) - zero_point)).astype(np.float32)
