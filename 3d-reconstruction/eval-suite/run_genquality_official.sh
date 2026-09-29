#!/bin/bash
# ==============================================================================
# SV3D-Eval-Suite : cross-method COMPARABLE generation-quality on the OFFICIAL
# MINE present split (test_files_present.txt, 3100 rows / 620 scenes).
#
# Why this script exists:
#   The stock evaluate.py of both repos names export dirs by scene hash only
#   (Flash3D: split("+")[1]) or by a mis-aligned scene list (CATSplat:
#   _seq_keys[k]). Both collapse/scramble the <=5 rows-per-scene, so the two
#   methods' image sets are NOT the same set -> cross-method FID is meaningless.
#
#   The patched evaluate.py (patches/*.py) names every dir by the split ROW:
#       {row:05d}_{scene_hash}_src{src_idx}
#   derived from dataset._seq_key_src_idx_pairs[k], which both repos iterate in
#   identical split-file order. Verified: both methods emit the SAME dir names
#   and per-dir GT is pixel-identical (GT-MAE = 0.000) -> FID is comparable.
#
# Output:
#   /home/data/E-052_official/flash3d_present_imgs2/<row_dir>/{pred,gt}/00X.png
#   /home/data/E-052_official/catsplat_present_imgs2/<row_dir>/{pred,gt}/00X.png
#   flash3d_genq_official.json / catsplat_genq_official.json  (FID/KID/DISTS/LPIPS)
# ==============================================================================
set -x
OFF=/home/data/E-052_official
SPLIT=splits/re10k_mine_filtered/test_files_present.txt
F3D_IMGS=$OFF/flash3d_present_imgs2
CAT_IMGS=$OFF/catsplat_present_imgs2

export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_VISIBLE_DEVICES=0
export SKIP_PLY=1
source /root/projects/flash3d/.venv/bin/activate

# ---------------------------------------------------------------- 1) Flash3D export
cd /root/projects/flash3d
mkdir -p $OFF/flash3d_run
ln -sfn /root/projects/flash3d/checkpoints $OFF/flash3d_run/checkpoints
rm -rf $F3D_IMGS
FLASH3D_VIS_DIR=$F3D_IMGS python -u evaluate.py \
  hydra.run.dir=$OFF/flash3d_run \
  hydra.job.chdir=false \
  +experiment=layered_re10k \
  +dataset.crop_border=true \
  dataset.test_split_path=$SPLIT \
  model.depth.version=v1 \
  ++eval.save_vis=true
echo "[gq-official] flash3d export done #dirs=$(ls $F3D_IMGS 2>/dev/null | wc -l)"

# ---------------------------------------------------------------- 2) CATSplat export
cd /root/projects/CATSplat
mkdir -p $OFF/catsplat_run
rm -rf $CAT_IMGS
CATSPLAT_VIS_DIR=$CAT_IMGS python -u evaluate.py \
  hydra.run.dir=$OFF/catsplat_run \
  hydra.job.chdir=false \
  +experiment=layered_re10k \
  +dataset.crop_border=true \
  dataset.data_path=/home/data/RealEstate10K \
  dataset.test_split_path=./$SPLIT \
  model.depth.version=v1 \
  ++eval.save_vis=true \
  run.checkpoint=/root/projects/CATSplat/ckpts/CATSplat.pth
echo "[gq-official] catsplat export done #dirs=$(ls $CAT_IMGS 2>/dev/null | wc -l)"

# ---------------------------------------------------------------- 3) gen-quality (cleanfid)
cd /root/projects/flash3d
python -u /home/data/sv3d-eval-suite/metrics/eval_genquality.py \
  --root "$F3D_IMGS" --method flash3d \
  --out $OFF/flash3d_genq_official.json --workdir $OFF/_genq_tmp2/flash3d
echo "[gq-official] flash3d genq done"

python -u /home/data/sv3d-eval-suite/metrics/eval_genquality.py \
  --root "$CAT_IMGS" --method catsplat \
  --out $OFF/catsplat_genq_official.json --workdir $OFF/_genq_tmp2/catsplat
echo "[gq-official] catsplat genq done"

echo "[gq-official] ALL DONE exit=$?"
