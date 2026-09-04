#!/bin/bash
set -e
export HF_HOME=/workspace/hf_cache
cd /workspace

echo "=== [1/3] Starting E1R-squash-448 ==="
/opt/venvs/anomaly/bin/python -u ad2_pixel_eval.py \
  --img 448 --bank-cap 4000 --geometry squash \
  --run-id E1R-squash-448 \
  --hypothesis "re-score squash under fixed native region set (E4a)" \
  --out outputs/runs/E1R-squash-448.json > /tmp/E1R-squash-448.log 2>&1
echo "=== [1/3] E1R-squash-448 completed ==="

echo "=== [2/3] Starting E2R-letterbox-448 ==="
/opt/venvs/anomaly/bin/python -u ad2_pixel_eval.py \
  --img 448 --bank-cap 4000 --geometry letterbox \
  --run-id E2R-letterbox-448 \
  --hypothesis "re-score letterbox under fixed native region set (E4a)" \
  --out outputs/runs/E2R-letterbox-448.json > /tmp/E2R-letterbox-448.log 2>&1
echo "=== [2/3] E2R-letterbox-448 completed ==="

echo "=== [3/3] Starting E3R-aspect-448 ==="
/opt/venvs/anomaly/bin/python -u ad2_pixel_eval.py \
  --img 448 --bank-cap 4000 --geometry aspect \
  --run-id E3R-aspect-448 \
  --hypothesis "under a fixed region set, E3's margin over E1 falls below 0.01 mean AU-PRO@5% and the two are within noise" \
  --out outputs/runs/E3R-aspect-448.json > /tmp/E3R-aspect-448.log 2>&1
echo "=== [3/3] E3R-aspect-448 completed ==="

echo "=== ALL E4a RUNS COMPLETED SUCCESSFULLY ==="
