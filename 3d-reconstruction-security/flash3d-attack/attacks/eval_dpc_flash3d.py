"""DPC attack against official Flash3D: white-box, source-camouflaged, pose-agnostic.

Threat model & novelty: see README / attacks/dpc_attack.py.

This wrapper:
  * loads official Flash3D (weights frozen),
  * builds a differentiable RenderFn(image, relative_pose) around the frozen model,
  * runs DPC on the source image using attacker-sampled AUXILIARY poses,
  * evaluates the attacked image on the UNSEEN MINE eval targets (tgt5/tgt10/tgt_rand),
  * reports source-view PSNR (camouflage) and novel-view PSNR/SSIM/LPIPS drops.

Run on server (Flash3D venv):
  cd /root/projects/flash3d && source .venv/bin/activate
  export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python /root/flash3d-attack/attacks/eval_dpc_flash3d.py --max_scenes 30 \
    --out /root/flash3d-attack/results/dpc_n30.json
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
from evaluation.evaluator import Evaluator
from datasets.util import create_datasets

from attacks.dpc_attack import (
    DPCConfig,
    dpc_attack,
    sample_auxiliary_poses,
    random_delta_baseline,
    naive_pgd_attack,
)
from attacks.dpc_blackbox import BlackBoxDPCConfig, blackbox_dpc_attack

FLASH3D_CKPT = "/root/projects/flash3d/checkpoints/model_re10k_v2.pth"
PAD_KEYS = "backproject_depth"


def _load(cfg, device):
    model = GaussianPredictor(cfg).to(device)
    state = torch.load(FLASH3D_CKPT, map_location="cpu")
    sd = state["model"] if "model" in state else state
    current = model.state_dict()
    filtered = {}
    for k, v in sd.items():
        if PAD_KEYS in k:
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
    """Differentiable render of `src_image` at one relative pose via the frozen model.

    We reuse Flash3D's forward to build Gaussians from the (attacked) source image, but
    override the target relative pose with `relative_pose` (attacker-chosen, constant).
    Returns rendered RGB (1,3,H,W) with grad flowing back to `src_image`.
    """
    local = dict(inputs)
    local[("color_aug", 0, 0)] = src_image
    local["target_frame_ids"] = [1]

    cfg = model.cfg
    outputs = model.models["unidepth_extended"](local)
    model.compute_gauss_means(local, outputs)

    # Build gaussians (opacity/scale/rotation/features) via the decoder path already run in
    # unidepth_extended forward; compute_gauss_means filled gauss_means. Now render one pose.
    B, _, H, W = src_image.shape
    device = src_image.device
    dtype = outputs["gauss_means"].dtype
    outputs[("cam_T_cam", 0, 1)] = (
        relative_pose.to(device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    )
    # Render just frame 1 by temporarily restricting frame ids.
    model.render_images(local, outputs)
    return outputs[("color_gauss", 1, 0)]


def _make_render_fn(model, inputs):
    def render_fn(image, pose):
        return _render_single(model, inputs, image, pose)

    return render_fn


def _psnr(pred, gt):
    mse = (pred.clamp(0, 1) - gt.clamp(0, 1)).pow(2).mean().item()
    if mse <= 0:
        return 99.0
    return -10.0 * torch.log10(torch.tensor(mse)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split", default="splits/re10k_mine_filtered/test_files_present.txt"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_scenes", type=int, default=30)
    ap.add_argument(
        "--method", choices=["dpc", "naive_pgd", "random", "blackbox"], default="dpc"
    )
    ap.add_argument("--epsilon", type=float, default=8.0 / 255.0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--step_size", type=float, default=1.0 / 255.0)
    ap.add_argument("--lambda_src", type=float, default=10.0)
    ap.add_argument("--n_aux_poses", type=int, default=4)
    ap.add_argument("--bb_iters", type=int, default=60)
    ap.add_argument("--bb_nes_samples", type=int, default=20)
    ap.add_argument("--bb_n_freq", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    cfg = _make_cfg(args.split)
    model = _load(cfg, device)
    evaluator = Evaluator(crop_border=cfg.dataset.crop_border).to(device)
    _, loader = create_datasets(cfg, split="test")

    dpc_cfg = DPCConfig(
        epsilon=args.epsilon,
        steps=args.steps,
        step_size=args.step_size,
        lambda_src=args.lambda_src,
        n_aux_poses=args.n_aux_poses,
    )

    agg = defaultdict(lambda: defaultdict(list))
    names = {0: "src", 1: "tgt5", 2: "tgt10", 3: "tgt_rand"}
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
        render_fn = _make_render_fn(model, inputs)
        if args.method == "dpc":
            attacked_src, delta, info = dpc_attack(clean_src, render_fn, dpc_cfg)
        elif args.method == "naive_pgd":
            attacked_src, delta, info = naive_pgd_attack(clean_src, render_fn, dpc_cfg)
        elif args.method == "blackbox":
            bb_cfg = BlackBoxDPCConfig(
                epsilon=args.epsilon,
                lambda_src=args.lambda_src,
                n_aux_poses=args.n_aux_poses,
                iters=args.bb_iters,
                nes_samples=args.bb_nes_samples,
                n_freq=args.bb_n_freq,
            )
            attacked_src, delta, info = blackbox_dpc_attack(
                clean_src, render_fn, bb_cfg
            )
        else:
            attacked_src, delta, info = random_delta_baseline(
                clean_src, dpc_cfg.epsilon, seed=dpc_cfg.seed
            )

        # Evaluate clean vs attacked on the UNSEEN eval targets using the full forward.
        # IMPORTANT: only the model input (`color_aug`) is perturbed. `color` stays the
        # untouched GT so metrics compare against the real reference images.
        for tag, src_img in [("clean", clean_src), ("attacked", attacked_src)]:
            eval_inputs = copy.deepcopy(inputs)
            eval_inputs[("color_aug", 0, 0)] = src_img.detach()
            eval_inputs["target_frame_ids"] = [1, 2, 3]
            with torch.no_grad():
                out = model(eval_inputs)
            for fid in [0, 1, 2, 3]:
                pred_key = ("color_gauss", fid, 0)
                gt_key = ("color", fid, 0)
                if pred_key not in out or gt_key not in eval_inputs:
                    continue
                pred = out[pred_key].clamp(0, 1)
                gt = eval_inputs[gt_key].clamp(0, 1)
                m = evaluator(pred, gt)
                for mk, mv in m.items():
                    agg[f"{tag}_{names[fid]}"][mk].append(float(mv))
        n_done += 1
        if n_done % 5 == 0:
            print(f"attacked {n_done} scenes; last linf={info['linf']:.4f}", flush=True)

    summary = {}
    for key, metrics in agg.items():
        summary[key] = {mk: float(sum(v) / len(v)) for mk, v in metrics.items() if v}
    summary["n_scenes"] = n_done
    summary["dpc_cfg"] = vars(args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
