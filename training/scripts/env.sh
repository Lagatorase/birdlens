#!/usr/bin/env bash
# I source this before running any training script inside WSL.
# TensorFlow 2.21 finds most of the pip-installed CUDA libraries by itself but
# missed libcusolver.so.11 on my machine, so I put every nvidia/*/lib folder
# from the venv on the library path.
WORK_DIR="${BIRDLENS_WORK:-$HOME/birdlens-work}"
export VIRTUAL_ENV="$WORK_DIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
NV_LIBS="$(find "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia" -maxdepth 2 -type d -name lib | tr '\n' ':')"
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"
# TF 2.21 has no prebuilt kernels for the RTX 5060 (compute capability 12.0),
# so CUDA compiles them from PTX. A bigger cache means that only happens once.
export CUDA_CACHE_MAXSIZE=4294967296
# I hide TensorFlow's info-level log spam but keep warnings.
export TF_CPP_MIN_LOG_LEVEL=1
