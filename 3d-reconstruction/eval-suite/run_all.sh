#!/bin/bash
# E-052 full eval orchestration. Waits for Flash3D export to finish, then runs
# external-metric driver on Flash3D imgs, gen-quality on both methods, and compiles.
# Uses cleanfid (local weight). Logs everything to /home/data/E-052_external/orch.log
set -x
cd /root/projects/flash3d
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_VISIBLE_DEVICES=0

EXT=/home/data/E-052_external
CAT="/root/projects/CATSplat/exp/evaluate/wide700/Put your ply file path"
F3D=$EXT/flash3d_wide700_imgs

# 1) wait for the Flash3D export process (evaluate.py) to finish
echo "[orch] waiting for flash3d export to finish..."
while pgrep -f "evaluate.py.*save_vis=true" >/dev/null 2>&1; do
  sleep 20
done
echo "[orch] export done. #scenes=$(ls $F3D 2>/dev/null | wc -l)"

# 2) external metrics on Flash3D exported imgs (official evaluator, 5% crop)
python -u _e052_eval_external.py --root "$F3D" --method flash3d_wide700 \
  --out $EXT/flash3d_wide700.json
echo "[orch] flash3d external done"

# 3) gen-quality (cleanfid FID/KID + DISTS + LPIPS) for both methods
python -u _e052_eval_genquality.py --root "$F3D" --method flash3d \
  --out $EXT/flash3d_genq.json --workdir $EXT/_genq_tmp/flash3d
echo "[orch] flash3d genq done"

python -u _e052_eval_genquality.py --root "$CAT" --method catsplat \
  --out $EXT/catsplat_genq.json --workdir $EXT/_genq_tmp/catsplat
echo "[orch] catsplat genq done"

# 4) compile comparison table
python -u _e052_compile_table.py \
  --recon flash3d=$EXT/flash3d_wide700.json \
  --recon catsplat=$EXT/catsplat_wide700.json \
  --genq flash3d=$EXT/flash3d_genq.json \
  --genq catsplat=$EXT/catsplat_genq.json \
  --out $EXT/COMPARISON.md
echo "[orch] ALL DONE exit=$?"
