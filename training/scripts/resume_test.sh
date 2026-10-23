#!/usr/bin/env bash
# I check that train.py really resumes mid-stage: start a short run in a
# throwaway folder, kill it during head epoch 3, start it again, and look at
# which epoch it continues from.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

TEST_DIR="$HOME/birdlens-resume-test"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR/logs" "$TEST_DIR/results"
export BIRDLENS_WORK="$TEST_DIR" BIRDLENS_LOGS="$TEST_DIR/logs" BIRDLENS_RESULTS="$TEST_DIR/results"

# First run: I kill it once the third head epoch has started.
python -u train.py --head-epochs 4 --finetune-epochs 1 > "$TEST_DIR/run1.log" 2>&1 &
PID=$!
until grep -q "Epoch 3/4" "$TEST_DIR/run1.log" 2>/dev/null; do
  if ! kill -0 "$PID" 2>/dev/null; then echo "run 1 ended early"; tail -5 "$TEST_DIR/run1.log"; exit 1; fi
  sleep 2
done
sleep 3
kill -9 "$PID"
echo "Killed run 1 during head epoch 3"
grep -E "^Epoch|val_accuracy" "$TEST_DIR/run1.log"

# Second run: it should restart at epoch 3, not epoch 1.
python -u train.py --head-epochs 4 --finetune-epochs 1 > "$TEST_DIR/run2.log" 2>&1
echo "Run 2:"
grep -E "^Epoch|already|Fine-tuning" "$TEST_DIR/run2.log"
echo "Head history rows (should be 4, one per epoch, no repeats):"
cut -d, -f1 "$TEST_DIR/logs/head_history.csv"
rm -rf "$TEST_DIR"
