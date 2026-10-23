"""Print the parameter count of the BirdLens model, for the README.

Usage: python scripts/count_params.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Counting parameters doesn't need the GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from birdlens_train.model import backbone_of, build_model  # noqa: E402

model = build_model(weights=None)
backbone = backbone_of(model)
print(f"total parameters: {model.count_params():,}")
print(f"backbone parameters: {backbone.count_params():,}")
print(f"head parameters: {model.count_params() - backbone.count_params():,}")
