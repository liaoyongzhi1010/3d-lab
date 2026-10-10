"""Gradient-flow probe: does d(novel_view_render)/d(source_rgb) exist and is it non-zero?

This is the decisive feasibility test for the white-box DPC attack. If UniDepth or the
rasterizer blocks gradients back to the input image, white-box DPC is not viable and we
must go pure black-box.

Run on server (Flash3D venv):
  cd /root/projects/flash3d && source .venv/bin/activate
  export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  python /root/flash3d-attack/attacks/probe_gradient.py 2>&1 | tail -30
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hydra import compose, initialize_config_dir
from models.model import GaussianPredictor
from datasets.util import create_datasets

FLASH3D_CKPT = "/root/projects/flash3d/checkpoints/model_re10k_v2.pth"


def _load(cfg, device):
    model = GaussianPredictor(cfg).to(device)
    state = torch.load(FLASH3D_CKPT, map_location="cpu")
    sd = state["model"] if "model" in state else state
    current = model.state_dict()
    filtered = {}
    for k, v in sd.items():
        if "backproject_depth" in k:
            if k in current:
                filtered[k] = current[k].clone()
        else:
            filtered[k] = v
    model.load_state_dict(filtered, strict=False)
    model.set_eval()
    return model


def main():
    device = torch.device("cuda:0")
    with initialize_config_dir(
        config_dir="/root/projects/flash3d/configs", version_base=None
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=layered_re10k",
                "+dataset.crop_border=true",
                "dataset.test_split_path=splits/re10k_mine_filtered/test_files_present.txt",
                "model.depth.version=v1",
                "data_loader.batch_size=1",
                "data_loader.num_workers=1",
            ],
        )
    model = _load(cfg, device)
    _, loader = create_datasets(cfg, split="test")
    inputs = next(iter(loader))
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    inputs["target_frame_ids"] = [1, 2, 3]

    # Make source RGB a leaf requiring grad.
    for key in [("color", 0, 0), ("color_aug", 0, 0)]:
        if key in inputs:
            inputs[key] = inputs[key].detach().clone().requires_grad_(True)

    src_leaf = inputs[("color_aug", 0, 0)]

    outputs = model(inputs)

    # novel-view render loss (target frame 1)
    novel = outputs[("color_gauss", 1, 0)]
    loss_novel = novel.pow(2).mean()
    grad_novel = torch.autograd.grad(
        loss_novel, src_leaf, retain_graph=True, allow_unused=True
    )[0]

    # source-view render loss (frame 0)
    src_render = outputs[("color_gauss", 0, 0)]
    loss_src = src_render.pow(2).mean()
    grad_src = torch.autograd.grad(
        loss_src, src_leaf, retain_graph=True, allow_unused=True
    )[0]

    print("=== GRADIENT FLOW PROBE ===")
    print(f"source leaf shape: {tuple(src_leaf.shape)}")
    if grad_novel is None:
        print("novel->src grad: NONE (blocked)")
    else:
        nz = (grad_novel.abs() > 0).float().mean().item()
        print(
            f"novel->src grad: nonzero_frac={nz:.4f} "
            f"abs_mean={grad_novel.abs().mean().item():.3e} abs_max={grad_novel.abs().max().item():.3e}"
        )
    if grad_src is None:
        print("src->src grad: NONE (blocked)")
    else:
        nz = (grad_src.abs() > 0).float().mean().item()
        print(
            f"src->src   grad: nonzero_frac={nz:.4f} "
            f"abs_mean={grad_src.abs().mean().item():.3e} abs_max={grad_src.abs().max().item():.3e}"
        )

    # Also check depth gradient path specifically.
    if grad_novel is not None and grad_src is not None:
        ratio = grad_novel.abs().mean().item() / (grad_src.abs().mean().item() + 1e-12)
        print(f"novel/src grad ratio (mean abs): {ratio:.3f}")
        print("VERDICT: white-box DPC is FEASIBLE (gradients flow to source RGB).")
    else:
        print("VERDICT: white-box DPC path BLOCKED; use black-box transfer.")


if __name__ == "__main__":
    main()
