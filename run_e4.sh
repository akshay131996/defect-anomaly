#!/bin/bash
set -e
export HF_HOME=/workspace/hf_cache
cd /workspace

mkdir -p outputs/runs logs

echo "=== [1/3] Starting E4-evalside-512 ==="
/opt/venvs/anomaly/bin/python -u ad2_pixel_eval.py \
  --img 448 --bank-cap 4000 --geometry aspect \
  --eval-side 512 --gauss-sigma 4.0 --resume \
  --run-id E4-evalside-512 \
  --hypothesis "mean AU-PRO@5% benchmark at nominal EVAL_SIDE=512 under winning aspect geometry" \
  --out outputs/runs/E4-evalside-512.json >> /tmp/E4-evalside-512.log 2>&1
cp /tmp/E4-evalside-512.log logs/E4-evalside-512.log
echo "=== [1/3] E4-evalside-512 completed ==="

echo "=== [2/3] Starting E4-evalside-1024 ==="
/opt/venvs/anomaly/bin/python -u ad2_pixel_eval.py \
  --img 448 --bank-cap 4000 --geometry aspect \
  --eval-side 1024 --gauss-sigma 8.0 --resume \
  --run-id E4-evalside-1024 \
  --hypothesis "mean AU-PRO@5% rises with EVAL_SIDE=1024 on high-region scenarios" \
  --out outputs/runs/E4-evalside-1024.json >> /tmp/E4-evalside-1024.log 2>&1
cp /tmp/E4-evalside-1024.log logs/E4-evalside-1024.log
echo "=== [2/3] E4-evalside-1024 completed ==="

echo "=== [3/3] Starting E4-evalside-2048 ==="
/opt/venvs/anomaly/bin/python -u ad2_pixel_eval.py \
  --img 448 --bank-cap 4000 --geometry aspect \
  --eval-side 2048 --gauss-sigma 16.0 --resume \
  --run-id E4-evalside-2048 \
  --hypothesis "mean AU-PRO@5% rises with EVAL_SIDE=2048 on high-region scenarios" \
  --out outputs/runs/E4-evalside-2048.json >> /tmp/E4-evalside-2048.log 2>&1
cp /tmp/E4-evalside-2048.log logs/E4-evalside-2048.log
echo "=== [3/3] E4-evalside-2048 completed ==="

echo "=== ALL E4 RUNS COMPLETED SUCCESSFULLY ==="
