#!/usr/bin/env bash
# I run this inside WSL2 to build the training environment.
# The venv and dataset live in my WSL home folder because reading thousands of
# small files through /mnt/c is slow, and I symlink them into training/ so the
# paths in the scripts stay simple.
set -euo pipefail

# I work out where the repo and the fast WSL work folder are.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="${BIRDLENS_WORK:-$HOME/birdlens-work}"
mkdir -p "$WORK_DIR"

# I use uv because Ubuntu 26.04 ships Python 3.14 and I need 3.11 without sudo.
UV="$HOME/.local/bin/uv"
if [ ! -x "$UV" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# I create the Python 3.11 venv in the WSL home folder.
"$UV" venv --python 3.11 "$WORK_DIR/.venv"

# I link it into the repo so training/.venv works like the brief describes.
ln -sfn "$WORK_DIR/.venv" "$REPO_DIR/training/.venv"

# I install the exact pinned versions.
VIRTUAL_ENV="$WORK_DIR/.venv" "$UV" pip install -r "$REPO_DIR/training/requirements.txt"

echo "Environment ready at $WORK_DIR/.venv"
