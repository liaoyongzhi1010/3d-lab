#!/bin/bash
cd /root/flash3d-attack
source /root/projects/flash3d/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUTROOT=experiments/WB-021-eps-sweep
mkdir -p $OUTROOT
# eps sweep at fixed lambda_src=10, 5 scenes, 40 steps: find setting with source<=3dB drop + geometry damage
for EPS in 0.007843 0.015686 0.023529; do
  python -u attacks/eval_dpc_geometry.py --mode hybrid --max_scenes 5 --steps 40 \
    --epsilon $EPS --lambda_src 10 \
    --out $OUTROOT/eps${EPS}.json
done
echo "EPS_SWEEP_DONE"
