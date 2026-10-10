#!/usr/bin/env bash
set -euo pipefail

CATDIR="/root/projects/catsplat"
DATADIR="/home/data/sv3d-lab"

echo "=== CATSplat Setup ==="

if [ -d "$CATDIR" ]; then
    echo "CATSplat already cloned at $CATDIR"
else
    echo "Cloning CATSplat..."
    git clone --recurse-submodules https://github.com/kuai-lab/iccv25_CATSplat.git "$CATDIR"
fi

cd "$CATDIR"

if [ ! -d ".venv" ]; then
    echo "Creating venv..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing torch (cu118)..."
pip install -q -r requirements-torch.txt --extra-index-url https://download.pytorch.org/whl/cu118

echo "Installing other deps..."
pip install -q -r requirements.txt

echo "Linking data..."
if [ ! -L "data" ] && [ ! -d "data" ]; then
    mkdir -p data
    ln -sf /home/data/RealEstate10K data/RealEstate10K
fi

echo "Downloading checkpoint..."
CKPT_DIR="$DATADIR/checkpoints/catsplat"
mkdir -p "$CKPT_DIR"
if [ ! -f "$CKPT_DIR/model_re10k.pth" ]; then
    echo "Attempting checkpoint download from CATSplat Synology NAS..."
    wget -q --no-check-certificate \
        "https://kuaicv.synology.me/weights/iccv2025/CatSplat/CATSplat" \
        -O "$CKPT_DIR/model_re10k.pth" || echo "WARN: Checkpoint download failed. Try manually."
fi

echo "=== CATSplat setup complete ==="
echo "To evaluate:"
echo "  cd $CATDIR && source .venv/bin/activate"
echo "  python evaluate.py hydra.run.dir=$DATADIR/evaluations/catsplat_repro \\"
echo "    hydra.job.chdir=true +experiment=layered_re10k +dataset.crop_border=true \\"
echo "    dataset.test_split_path=splits/re10k_mine_filtered/test_files.txt \\"
echo "    model.depth.version=v1 ++eval.save_vis=false \\"
echo "    run.checkpoint=$CKPT_DIR/model_re10k.pth"
