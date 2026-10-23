"""Shared settings so every script agrees on paths and sizes."""

import os
from pathlib import Path

# I resolve paths from this file so the scripts work from any folder.
TRAINING_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = TRAINING_DIR.parent

# The dataset and checkpoints live in the WSL home folder for speed, and the
# training/data link points there. I allow an override for tests.
DATA_ROOT = Path(os.environ.get("BIRDLENS_DATA", TRAINING_DIR / "data" / "CUB_200_2011"))
WORK_DIR = Path(os.environ.get("BIRDLENS_WORK", Path.home() / "birdlens-work"))
CHECKPOINT_DIR = WORK_DIR / "checkpoints"
EXPORT_DIR = WORK_DIR / "exports"

# These are committed so the split and results can be reproduced.
SPLITS_DIR = TRAINING_DIR / "splits"
# Both can be redirected, which I use to test resuming without touching the
# real logs and results.
RESULTS_DIR = Path(os.environ.get("BIRDLENS_RESULTS", TRAINING_DIR / "results"))
LOG_DIR = Path(os.environ.get("BIRDLENS_LOGS", TRAINING_DIR / "logs"))
DOCS_DIR = REPO_DIR / "docs"
APP_ASSETS_DIR = REPO_DIR / "app" / "assets"
APP_FIXTURES_DIR = REPO_DIR / "app" / "test" / "fixtures"

# Model input size. The app reads the same number from the model file.
IMG_SIZE = 224
NUM_CLASSES = 200

# One seed for the validation split and for shuffling.
SEED = 42
VAL_FRACTION = 0.10

# Light GPU mode: I cap TensorFlow at roughly half of the 8151 MiB RTX 5060.
GPU_MEMORY_LIMIT_MB = 4000
BATCH_SIZE = 16
# I keep data loading workers modest so the game keeps its CPU time.
DATA_WORKERS = 4
