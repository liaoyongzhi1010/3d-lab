"""Evaluate structured non-PGD attacks against official Flash3D (standalone).

This file lives outside sv3d-lab and does NOT modify it. It loads the official
Flash3D checkpoint (weights fixed), attacks the input batch (source RGB and/or
camera metadata), runs forward(), and reports the same Evaluator metrics.

Run on server (Flash3D venv):
  cd /root/projects/flash3d && source .venv/bin/activate
  export CUDA_HOME=/usr/local/cuda-11.8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python /root/flash3d-attack/attacks/eval_flash3d_attack.py \
    --attack camera_drift --max_batches 100 \
    --out /root/flash3d-attack/results/attack_camera_drift_n100.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from contextlib import nullcontext

import torch

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hydra import compose, initialize_config_dir
from models.model import GaussianPredictor
from evaluation.evaluator import Evaluator
from datasets.util import create_datasets

from attacks.attack_transforms import (
    CameraDriftConfig,
    DepthCueTrapConfig,
    apply_camera_preprocess_drift,
    make_depth_cue_trap,
)

FLASH3D_CKPT = "/root/projects/flash3d/checkpoints/model_re10k_v2.pth"


def _to_device(inputs, device):
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)
    return inputs


def _load_official_flash3d(cfg, device):
    model = GaussianPredictor(cfg).to(device)
    state = torch.load(FLASH3D_CKPT, map_location="cpu")
    sd = state["model"] if "model" in state else state
    current = model.state_dict()
    filtered = {}
    for key, value in sd.items():
        if "backproject_depth" in key:
            if key in current:
                filtered[key] = current[key].clone()
        else:
            filtered[key] = value
    result = model.load_state_dict(filtered, strict=False)
    non_unidepth_missing = [k for k in result.missing_keys if "unidepth" not in k]
    if non_unidepth_missing or result.unexpected_keys:
        print(
            "load_state_dict warning: "
            f"non_unidepth_missing={len(non_unidepth_missing)} "
            f"unexpected={len(result.unexpected_keys)}"
        )
    model.set_eval()
    return model


def _make_cfg(split_path, batch_size):
    with initialize_config_dir(
        config_dir="/root/projects/flash3d/configs", version_base=None
    ):
        return compose(
            config_name="config",
            overrides=[
                "+experiment=layered_re10k",
                "+dataset.crop_border=true",
                f"dataset.test_split_path={split_path}",
                "model.depth.version=v1",
                f"data_loader.batch_size={batch_size}",
                "data_loader.num_workers=4",
                "++eval.save_vis=false",
            ],
        )


def _apply_attack(inputs, attack, args):
    if attack == "none":
        return inputs
    if attack == "camera_drift":
        return apply_camera_preprocess_drift(
            inputs,
            CameraDriftConfig(
                focal_scale=args.focal_scale,
                pp_shift_px=(args.pp_shift_x, args.pp_shift_y),
                yaw_deg=args.yaw_deg,
                pitch_deg=args.pitch_deg,
                roll_deg=args.roll_deg,
                translation_scale=args.translation_scale,
            ),
        )
    if attack == "depth_cue_trap":
        attacked = {
            key: (value.clone() if isinstance(value, torch.Tensor) else value)
            for key, value in inputs.items()
        }
        for color_key in [("color", 0, 0), ("color_aug", 0, 0)]:
            if color_key in attacked and isinstance(attacked[color_key], torch.Tensor):
                attacked[color_key], _ = make_depth_cue_trap(
                    attacked[color_key],
                    DepthCueTrapConfig(
                        strength=args.trap_strength,
                        stripe_period=args.stripe_period,
                        shadow_width=args.shadow_width,
                        edge_blur=args.edge_blur,
                    ),
                )
        return attacked
    if attack == "combined":
        drifted = _apply_attack(inputs, "camera_drift", args)
        return _apply_attack(drifted, "depth_cue_trap", args)
    raise ValueError(f"Unknown attack: {attack}")


def _mean_scores(raw):
    out = {}
    for frame_id, metrics in raw.items():
        name = metrics.get("name", str(frame_id))
        out[name] = {}
        for metric_name, values in metrics.items():
            if metric_name == "name":
                continue
            if values:
                out[name][metric_name] = float(sum(values) / len(values))
    for metric in ["psnr", "ssim", "lpips"]:
        vals = [v[metric] for k, v in out.items() if k != "src" and metric in v]
        if vals:
            out.setdefault("target_avg", {})[metric] = float(sum(vals) / len(vals))
    return out


def evaluate_attack(model, evaluator, loader, device, attack, args):
    scores = defaultdict(lambda: defaultdict(list))
    frame_names = {0: "src", 1: "tgt5", 2: "tgt10", 3: "tgt_rand"}
    context = nullcontext() if args.whitebox else torch.no_grad()
    with context:
        for batch_idx, inputs in enumerate(loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            inputs = _to_device(inputs, device)
            inputs["target_frame_ids"] = [1, 2, 3]
            attacked_inputs = _apply_attack(inputs, attack, args)
            attacked_inputs["target_frame_ids"] = [1, 2, 3]
            outputs = model(attacked_inputs)
            for frame_id in model.all_frame_ids(attacked_inputs):
                gt_key = ("color", frame_id, 0)
                pred_key = ("color_gauss", frame_id, 0)
                if gt_key not in attacked_inputs or pred_key not in outputs:
                    continue
                pred = outputs[pred_key].clamp(0, 1)
                gt = attacked_inputs[gt_key].clamp(0, 1)
                metrics = evaluator(pred, gt)
                scores[frame_id]["name"] = frame_names.get(frame_id, str(frame_id))
                for key, value in metrics.items():
                    scores[frame_id][key].append(float(value))
            if (batch_idx + 1) % 25 == 0:
                print(f"processed {batch_idx + 1} batches", flush=True)
    return _mean_scores(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attack",
        choices=["none", "camera_drift", "depth_cue_trap", "combined"],
        required=True,
    )
    parser.add_argument(
        "--split", default="splits/re10k_mine_filtered/test_files_present.txt"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument(
        "--whitebox",
        action="store_true",
        help="Keep gradients enabled for future white-box variants.",
    )
    parser.add_argument("--focal_scale", type=float, default=1.04)
    parser.add_argument("--pp_shift_x", type=float, default=4.0)
    parser.add_argument("--pp_shift_y", type=float, default=-3.0)
    parser.add_argument("--yaw_deg", type=float, default=0.75)
    parser.add_argument("--pitch_deg", type=float, default=0.0)
    parser.add_argument("--roll_deg", type=float, default=0.0)
    parser.add_argument("--translation_scale", type=float, default=1.0)
    parser.add_argument("--trap_strength", type=float, default=0.08)
    parser.add_argument("--stripe_period", type=int, default=12)
    parser.add_argument("--shadow_width", type=int, default=21)
    parser.add_argument("--edge_blur", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    cfg = _make_cfg(args.split, args.batch_size)
    model = _load_official_flash3d(cfg, device)
    evaluator = Evaluator(crop_border=cfg.dataset.crop_border).to(device)
    _, loader = create_datasets(cfg, split="test")
    summary = evaluate_attack(model, evaluator, loader, device, args.attack, args)
    summary["attack"] = vars(args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
