#!/bin/bash
# BB-002: Real VGGT transfer attack (Flash3D surrogate -> VGGT target, 30 scenes)
set -e

cd /root/projects/flash3d && source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "=== BB-002: VGGT Transfer (30 scenes, eps=4/255, 40 steps) $(date) ==="
python /root/flash3d-attack/eval_vggt_transfer.py \
    --max_scenes 30 \
    --epsilon 0.015686 \
    --steps 40 \
    --lambda_src 3.0 \
    --out /root/flash3d-attack/results/BB-002-vggt-transfer.json

echo "=== DONE $(date) ==="
