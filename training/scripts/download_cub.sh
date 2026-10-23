#!/usr/bin/env bash
# I download CUB-200-2011 from the official CaltechDATA record and check its MD5
# before extracting, so I know the archive arrived intact.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="${BIRDLENS_WORK:-$HOME/birdlens-work}"
DATA_DIR="$WORK_DIR/data"
URL="https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
MD5="97eceeb196236b17998738112f37df78"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# I skip the download if a verified archive is already here.
if [ ! -f CUB_200_2011.tgz ] || ! echo "$MD5  CUB_200_2011.tgz" | md5sum -c --quiet; then
  # I resume partial downloads with -C - in case the connection drops.
  curl -L --fail --retry 5 -C - -o CUB_200_2011.tgz "$URL"
  echo "$MD5  CUB_200_2011.tgz" | md5sum -c
fi

# I extract only once.
if [ ! -f CUB_200_2011/train_test_split.txt ]; then
  tar -xzf CUB_200_2011.tgz
fi

# I link the data folder into the repo, where it is gitignored.
ln -sfn "$DATA_DIR" "$REPO_DIR/training/data"

echo "Images: $(find CUB_200_2011/images -name '*.jpg' | wc -l)"
