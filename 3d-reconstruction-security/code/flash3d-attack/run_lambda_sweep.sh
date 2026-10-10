#!/bin/bash
set -e
cd /root/flash3d-attack
source /root/projects/flash3d/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p experiments/WB-020-lambda-sweep
for LAM in 30 60 100; do
  python -u attacks/eval_dpc_geometry.py --mode hybrid --max_scenes 5 --steps 40 \
    --epsilon 0.031372549 --lambda_src $LAM \
    --out experiments/WB-020-lambda-sweep/lam${LAM}.json
done
echo "SWEEP_DONE"
