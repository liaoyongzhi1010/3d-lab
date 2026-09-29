"""Defense probe: can a test-time multi-view consistency check DETECT DPC?

Intuition: DPC corrupts parallax-dependent geometry. A defender without ground truth can still
measure INTERNAL consistency: render the reconstruction at a small pose and at a slightly larger
pose in the same direction; for a correct 3D scene these are geometrically consistent (warp-based
photometric residual is small), whereas DPC inflates the residual because the geometry is broken.

We compute a training-free consistency score S(x) = mean over probe directions of the
photometric residual between R_{a*P}(x) and R_{P}(x) after aligning via the model's own depth is
hard; instead we use the simplest detector: variance/energy of the multi-pose render stack around
the source. We then report detection AUC separating clean vs DPC-attacked inputs.

This is a DETECTOR (not a fix): it shows DPC is at least partially detectable, and quantifies how
much, which is the honest thing to report.

Run on server (Flash3D venv):
  python /root/flash3d-attack/attacks/eval_defense.py --max_scenes 40 \
    --out /root/flash3d-attack/results/defense_n40.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, "/root/projects/flash3d")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hydra import compose, initialize_config_dir
from models.model import GaussianPredictor
from datasets.util import create_datasets

from attacks.dpc_attack import DPCConfig, dpc_attack, sample_auxiliary_poses

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
    device = src_image.device
    dtype = outputs["gauss_means"].dtype
    outputs[("cam_T_cam", 0, 1)] = (
        relative_pose.to(device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    )
    model.render_images(local, outputs)
    return outputs[("color_gauss", 1, 0)]


def _small_pose(yaw_deg, tx, device):
    ry = torch.deg2rad(torch.tensor(yaw_deg))
    cy, sy = torch.cos(ry), torch.sin(ry)
    T = torch.eye(4, device=device)
    T[0, 0] = cy
    T[0, 2] = sy
    T[2, 0] = -sy
    T[2, 2] = cy
    T[0, 3] = tx
    return T


def consistency_score(model, inputs, src_image, device):
    """Training-free internal-consistency score. Higher = more likely attacked.

    For each probe direction, compare render at pose P vs at pose 2P. A consistent 3D scene
    changes smoothly; broken geometry produces large second-order photometric jumps. We use the
    residual || (R_2P - R_P) - (R_P - R_0) || (discrete curvature of the view manifold).
    """
    directions = [(3.0, 0.03), (-3.0, -0.03), (0.0, 0.05), (0.0, -0.05)]
    with torch.no_grad():
        r0 = _render_single(model, inputs, src_image, torch.eye(4, device=device))
        score = 0.0
        for yaw, tx in directions:
            P = _small_pose(yaw, tx, device)
            P2 = _small_pose(2 * yaw, 2 * tx, device)
            rP = _render_single(model, inputs, src_image, P)
            r2P = _render_single(model, inputs, src_image, P2)
            curvature = ((r2P - rP) - (rP - r0)).pow(2).mean().item()
            score += curvature
    return score / len(directions)


def _auc(clean_scores, attacked_scores):
    # attacked expected to have HIGHER score
    pos = attacked_scores
    neg = clean_scores
    wins = 0
    total = 0
    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / total if total else float("nan")


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

    clean_scores, attacked_scores = [], []
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

        clean_scores.append(consistency_score(model, inputs, clean_src, device))
        attacked_scores.append(
            consistency_score(model, inputs, attacked_src.detach(), device)
        )
        n_done += 1
        if n_done % 5 == 0:
            print(f"scored {n_done} scenes", flush=True)

    auc = _auc(clean_scores, attacked_scores)
    summary = {
        "n_scenes": n_done,
        "detector_auc": auc,
        "clean_score_mean": sum(clean_scores) / len(clean_scores)
        if clean_scores
        else None,
        "attacked_score_mean": sum(attacked_scores) / len(attacked_scores)
        if attacked_scores
        else None,
        "epsilon": args.epsilon,
        "lambda_src": args.lambda_src,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
