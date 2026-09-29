#!/bin/bash
set -e
cd /root/flash3d-attack
source /root/projects/flash3d/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p experiments/WB-010-baseline-calibration
python -u attacks/eval_dpc_geometry.py --mode hybrid --max_scenes 1 --steps 3 --epsilon 0.031372549 \
  --out experiments/WB-010-baseline-calibration/hybrid_smoke1.json
