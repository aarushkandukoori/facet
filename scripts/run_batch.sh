#!/bin/bash
# Usage: run_batch.sh <queue_file> <max_parallel>
# Each line of queue_file is a full argument string for train_ppo.py.
cd "$(dirname "$0")/.."
export PYTHONPATH=src:vendor/overcooked_ai/src
Q="$1"; P="${2:-4}"
cat "$Q" | xargs -P "$P" -I {} bash -c '
  line="$1"
  tag=$(echo "$line" | tr " /" "__")
  echo "[start $(date +%H:%M:%S)] $line"
  .venv/bin/python scripts/train_ppo.py $line > "logs/run_${tag}.log" 2>&1
  rc=$?
  echo "[done  $(date +%H:%M:%S)] $line (exit $rc)"
' _ {}
echo "BATCH_COMPLETE $Q"
