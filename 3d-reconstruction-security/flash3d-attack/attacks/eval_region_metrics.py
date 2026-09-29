"""Region-separated evaluation of DPC damage.

Where does DPC's corruption land? A geometry-level attack should concentrate damage in the parts
of a novel view that the source camera never observed (disoccluded / beyond-frustum), not in the
directly-visible region. We test this by partitioning every novel view into

    visible  : re-projected from the source (front-most warp),
    occluded : in-frame disocclusion holes behind foreground,
    beyond   : outside the source frustum coverage,

using ONLY the source depth + camera poses (method-agnostic; see attacks/region_masks.py). For each
region we report clean vs attacked PSNR and their drop. The masks are computed from the CLEAN source
depth so the clean/attacked comparison uses identical regions.

Run on server (flash3d venv):
  python /root/flash3d-attack/attacks/eval_region_metrics.py --max_scenes 60 \
    --out /root/flash3d-attack/results/region_metrics_n60.json
"""

from __future__ import annotations

import argparse
import copy
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

from attacks.dpc_attack import (
    DPCConfig,
    dpc_attack,
    random_delta_baseline,
    naive_pgd_attack,
)
from attacks.region_masks import region_partition, masked_psnr

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


def _unpad_depth(out, pad):
    """(B*gpp,1,Hpad,Wpad) -> (H,W) first layer, unpadded."""
    dp = out[("depth", 0)][0, 0]
    if pad:
        dp = dp[pad : dp.shape[0] - pad, pad : dp.shape[1] - pad]
    return dp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split", default="splits/re10k_mine_filtered/test_files_present.txt"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_scenes", type=int, default=60)
    ap.add_argument("--method", choices=["dpc", "naive_pgd", "random"], default="dpc")
    ap.add_argument("--epsilon", type=float, default=8.0 / 255.0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lambda_src", type=float, default=3.0)
    ap.add_argument("--n_aux_poses", type=int, default=4)
    ap.add_argument("--min_region_frac", type=float, default=0.02)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    cfg = _make_cfg(args.split)
    pad = cfg.dataset.pad_border_aug
    model = _load(cfg, device)
    _, loader = create_datasets(cfg, split="test")
    dpc_cfg = DPCConfig(
        epsilon=args.epsilon,
        steps=args.steps,
        lambda_src=args.lambda_src,
        n_aux_poses=args.n_aux_poses,
    )

    names = {1: "tgt5", 2: "tgt10", 3: "tgt_rand"}
    regions = ["visible", "occluded", "beyond"]
    # agg[region] -> list of (clean_psnr, attacked_psnr) pairs
    agg = {r: {"clean": [], "attacked": [], "frac": []} for r in regions}
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

        if args.method == "dpc":
            attacked_src, _, _ = dpc_attack(clean_src, render_fn, dpc_cfg)
        elif args.method == "naive_pgd":
            attacked_src, _, _ = naive_pgd_attack(clean_src, render_fn, dpc_cfg)
        else:
            attacked_src, _, _ = random_delta_baseline(
                clean_src, dpc_cfg.epsilon, seed=dpc_cfg.seed
            )

        # Full forward for clean (also gives depth + poses for masks) and attacked.
        clean_inputs = copy.deepcopy(inputs)
        clean_inputs[("color_aug", 0, 0)] = clean_src
        clean_inputs["target_frame_ids"] = [1, 2, 3]
        att_inputs = copy.deepcopy(inputs)
        att_inputs[("color_aug", 0, 0)] = attacked_src.detach()
        att_inputs["target_frame_ids"] = [1, 2, 3]
        with torch.no_grad():
            clean_out = model(clean_inputs)
            att_out = model(att_inputs)

        depth_src = _unpad_depth(clean_out, pad)
        H, W = depth_src.shape
        K = inputs[("K_tgt", 0)][0].float()

        for fid in [1, 2, 3]:
            pk = ("color_gauss", fid, 0)
            gk = ("color", fid, 0)
            if pk not in clean_out or gk not in inputs:
                continue
            gt = inputs[gk][0].clamp(0, 1)
            r_clean = clean_out[pk][0].clamp(0, 1)
            r_att = att_out[pk][0].clamp(0, 1)
            T = clean_out[("cam_T_cam", 0, fid)][0].float()
            vis, occ, beyond = region_partition(depth_src, K, T, H, W, device)
            for rname, mask in zip(regions, [vis, occ, beyond]):
                frac = float(mask.float().mean())
                if frac < args.min_region_frac:
                    continue
                pc = masked_psnr(r_clean, gt, mask)
                pa = masked_psnr(r_att, gt, mask)
                if pc is None or pa is None:
                    continue
                agg[rname]["clean"].append(pc)
                agg[rname]["attacked"].append(pa)
                agg[rname]["frac"].append(frac)

        n_done += 1
        if n_done % 5 == 0:
            print(f"processed {n_done} scenes", flush=True)

    summary = {"regions": {}}
    for r in regions:
        c = agg[r]["clean"]
        a = agg[r]["attacked"]
        if not c:
            summary["regions"][r] = {"n": 0}
            continue
        cm = sum(c) / len(c)
        am = sum(a) / len(a)
        summary["regions"][r] = {
            "clean_psnr": cm,
            "attacked_psnr": am,
            "delta_psnr": am - cm,
            "mean_frac": sum(agg[r]["frac"]) / len(agg[r]["frac"]),
            "n": len(c),
        }
    summary["n_scenes"] = n_done
    summary["config"] = vars(args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
