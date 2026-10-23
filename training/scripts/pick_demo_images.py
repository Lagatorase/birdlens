"""Pick a few CUB test images to push to the emulator gallery for screenshots.

I take three confidently correct photos of different-looking species and one
photo where the model falls below the threshold, so the "not confident"
message can be shown honestly with a real case.

Usage: python scripts/pick_demo_images.py --variant float16 --out <folder>
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from birdlens_train.config import DATA_ROOT, EXPORT_DIR, RESULTS_DIR, SPLITS_DIR  # noqa: E402
from birdlens_train.dataset import read_split_csv  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--variant", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

names = (SPLITS_DIR / "labels.txt").read_text(encoding="utf-8").splitlines()
rows = read_split_csv(SPLITS_DIR / "test.csv")
probs = np.load(EXPORT_DIR / f"test_probs_{args.variant}.npy")
labels = np.load(EXPORT_DIR / "test_labels.npy")
threshold = json.loads((RESULTS_DIR / "threshold.json").read_text())["threshold"]
conf, pred = probs.max(axis=1), probs.argmax(axis=1)

# Species I think make for clear, colourful screenshots.
wanted = ["Painted Bunting", "Blue Jay", "Cardinal", "Scarlet Tanager"]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
picked = []
for name in wanted:
    c = names.index(name)
    idx = np.where((labels == c) & (pred == c) & (conf > 0.7))[0]
    if len(idx):
        picked.append((int(idx[0]), "confident"))
# One real low-confidence case, wrong or right, under the app's threshold.
low = np.where(conf < threshold * 0.8)[0]
picked.append((int(low[0]), "low"))

for n, (i, kind) in enumerate(picked, start=1):
    src = DATA_ROOT / "images" / rows[i][0]
    dst = out / f"bird_{n}.jpg"
    shutil.copy(src, dst)
    print(f"{dst.name}: true={names[labels[i]]} pred={names[pred[i]]} conf={conf[i]:.2f} ({kind})")
