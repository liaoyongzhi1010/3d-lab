"""Cross-architecture evidence via the SHARED depth prior.

CATSplat native inference needs assets the public repo does not ship (per-scene LLaVA features +
a pointnet head with a non-standard return), so a full second-model pipeline is out of scope here.
Instead we test the *mechanism* directly: DPC perturbations optimized ONLY through Flash3D's
novel-view rendering loss should also corrupt the underlying UniDepth monocular-depth prediction,
which is the component shared across single-image 3DGS methods (Flash3D, CATSplat, and any
UniDepth-based reconstructor). If the raw depth map moves a lot while the source RGB moves little,
the attack is hitting a shared, architectural component — not a Flash3D-specific rendering quirk.

Metrics per scene (clean vs DPC-attacked source):
  - depth_absrel : mean |d_att - d_clean| / d_clean  (relative depth change)
  - depth_delta1 : fraction of pixels with max(d_att/d_clean, d_clean/d_att) > 1.10 (>10% shift)
  - src_rgb_psnr : PSNR between attacked and clean SOURCE image (camouflage sanity)

Run on server (flash3d venv):
  python /root/flash3d-attack/attacks/eval_depth_transfer.py --max_scenes 40 \
    --out /root/flash3d-attack/results/depth_transfer_n40.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hydra import compose, initialize_config_dir
from models.model import GaussianPredictor
from datasets.util import create_datasets

from attacks.dpc_attack import DPCConfig, dpc_attack

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


def _make_cfg(split):
    with initialize_config_dir(
        config_dir="/root/projects/flash3d/configs", version_base=None
    ):
        return compose(
            config_name="config",
            overrides=[
                "+experiment=layered_re10k",
                "+dataset.crop_border=true",
                f"dataset.test_split_path={split}",
                "model.depth.version=v1",
                "data_loader.batch_size=1",
                "data_loader.num_workers=1",
            ],
        )


def _render_single(model, inputs, src_image, relative_pose):
    local = dict(inputs)
    local[("color_aug", 0, 0)] = src_image
    local["target_frame_ids"] = [1]
    outputs = model.models["unidepth_extended"](local)
    model.compute_gauss_means(local, outputs)
    B = src_image.shape[0]
    dtype = outputs["gauss_means"].dtype
    outputs[("cam_T_cam", 0, 1)] = (
        relative_pose.to(device=src_image.device, dtype=dtype)
        .unsqueeze(0)
        .repeat(B, 1, 1)
    )
    model.render_images(local, outputs)
    return outputs[("color_gauss", 1, 0)]


def _depth_of(model, inputs, src_image):
    """Return the UniDepth depth map for a given source image (no grad)."""
    local = dict(inputs)
    local[("color_aug", 0, 0)] = src_image
    local["target_frame_ids"] = [1]
    with torch.no_grad():
        outputs = model.models["unidepth_extended"](local)
    scale = model.cfg.model.scales[0]
    return outputs[("depth", scale)].detach()


def _psnr(a, b):
    mse = (a.clamp(0, 1) - b.clamp(0, 1)).pow(2).mean().item()
    if mse <= 0:
        return 99.0
    return -10.0 * torch.log10(torch.tensor(mse)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split", default="splits/re10k_mine_filtered/test_files_present.txt"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_scenes", type=int, default=40)
    ap.add_argument("--epsilon", type=float, default=8.0 / 255.0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lambda_src", type=float, default=3.0)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    cfg = _make_cfg(args.split)
    model = _load(cfg, device)
    _, loader = create_datasets(cfg, split="test")
    dpc_cfg = DPCConfig(
        epsilon=args.epsilon, steps=args.steps, lambda_src=args.lambda_src
    )

    agg = defaultdict(list)
    n_done = 0
    for inputs in loader:
        if n_done >= args.max_scenes:
            break
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        if ("color", 1, 0) not in inputs:
            continue
        inputs["target_frame_ids"] = [1, 2, 3]
        clean_src = inputs[("color_aug", 0, 0)].detach()

        def render_fn(image, pose):
            return _render_single(model, inputs, image, pose)

        attacked_src, _, _ = dpc_attack(clean_src, render_fn, dpc_cfg)

        d_clean = _depth_of(model, inputs, clean_src)
        d_att = _depth_of(model, inputs, attacked_src.detach())

        eps = 1e-6
        absrel = ((d_att - d_clean).abs() / (d_clean.abs() + eps)).mean().item()
        ratio = torch.maximum(d_att / (d_clean + eps), d_clean / (d_att + eps))
        delta1 = (ratio > 1.10).float().mean().item()
        src_psnr = _psnr(attacked_src.detach(), clean_src)

        agg["depth_absrel"].append(absrel)
        agg["depth_delta_gt10pct"].append(delta1)
        agg["src_rgb_psnr"].append(src_psnr)
        n_done += 1
        if n_done % 5 == 0:
            print(
                f"{n_done} scenes; absrel={absrel:.3f} delta>10%={delta1:.3f} srcPSNR={src_psnr:.1f}",
                flush=True,
            )

    summary = {k: float(sum(v) / len(v)) for k, v in agg.items() if v}
    summary["n_scenes"] = n_done
    summary["config"] = vars(args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
