#!/bin/bash
cd /root/flash3d-attack
source /root/projects/flash3d/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=experiments/WB-031-pilot-eps4
mkdir -p $OUT
# eps=4/255 operating point, 30 scenes, 40 steps; hybrid geometry attack
python -u attacks/eval_dpc_geometry.py --mode hybrid --max_scenes 30 --steps 40 \
  --epsilon 0.015686 --lambda_src 10 --out $OUT/results.json
# baselines at same eps for comparison
python -u attacks/eval_dpc_geometry.py --mode primitive --max_scenes 30 --steps 40 \
  --epsilon 0.015686 --lambda_src 10 --out $OUT/primitive.json
echo "PILOT_EPS4_DONE"
